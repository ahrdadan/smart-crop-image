package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultPort                  = "8080"
	defaultJobQueueCapacity      = 1000
	defaultJobTTLHours           = 2
	defaultJobProcessTimeoutSecs = 1800
	defaultImageQuality          = 95
	defaultWebhookTimeoutSecs    = 15
	defaultWebhookRetries        = 3
	defaultWebhookBackoffMs      = 2000
	defaultInlineImageMaxBytes   = 20 * 1024 * 1024
	defaultInlineImageMaxCount   = 50
)

type ThumbnailRequest struct {
	ImagePath        string      `json:"image_path"`
	ImagePaths       []string    `json:"image_paths"`
	ImageFiles       []imageFile `json:"image_files"`
	OutputPath       string      `json:"output_path"`
	ReturnCandidates bool        `json:"return_candidates"`
	ComposeMode      string      `json:"compose_mode"`
	PreferredRatio   interface{} `json:"preferred_ratio"`
	MaxAnalysisSize  int         `json:"max_analysis_size"`
	ApplyCrop        bool        `json:"apply_crop"`
	Quality          int         `json:"quality"`
	WebhookURL       string      `json:"webhook_url"`
}

type imageFile struct {
	Filename      string `json:"filename"`
	ContentBase64 string `json:"content_base64"`
}

type CropResult struct {
	CropX      int     `json:"crop_x"`
	CropY      int     `json:"crop_y"`
	CropWidth  int     `json:"crop_width"`
	CropHeight int     `json:"crop_height"`
	Method     string  `json:"method"`
	Confidence float64 `json:"confidence"`
}

type ThumbnailCandidate struct {
	PageIndex  int     `json:"page_index"`
	ImagePath  string  `json:"image_path"`
	CropX      int     `json:"crop_x"`
	CropY      int     `json:"crop_y"`
	CropWidth  int     `json:"crop_width"`
	CropHeight int     `json:"crop_height"`
	Method     string  `json:"method"`
	Confidence float64 `json:"confidence"`
	Score      float64 `json:"score"`
}

type ThumbnailResponse struct {
	CropResult
	Applied           bool                 `json:"applied"`
	FallbackUsed      bool                 `json:"fallback_used,omitempty"`
	OutputPath        string               `json:"output_path,omitempty"`
	WorkerWarning     string               `json:"worker_warning,omitempty"`
	CropError         string               `json:"crop_error,omitempty"`
	CompositionMode   string               `json:"composition_mode,omitempty"`
	ComposedFrom      []string             `json:"composed_from,omitempty"`
	SelectedPageIndex *int                 `json:"selected_page_index,omitempty"`
	SelectedImagePath string               `json:"selected_image_path,omitempty"`
	SelectedScore     float64              `json:"selected_score,omitempty"`
	Candidates        []ThumbnailCandidate `json:"candidates,omitempty"`
}

type smartPairPicked struct {
	Idx       int     `json:"idx"`
	Path      string  `json:"path"`
	Total     float64 `json:"total"`
	TextRatio float64 `json:"text_ratio"`
	InEdge    bool    `json:"in_edge"`
	BBox      []int   `json:"bbox"`
}

type smartPairResult struct {
	OutPath        string            `json:"out_path"`
	Size           []int             `json:"size"`
	Picked         []smartPairPicked `json:"picked"`
	FallbackUsed   bool              `json:"fallback_used,omitempty"`
	FallbackReason string            `json:"fallback_reason,omitempty"`
}

type commandCandidate struct {
	Bin  string
	Args []string
}

type jobStatus string

const (
	jobQueued     jobStatus = "queued"
	jobProcessing jobStatus = "processing"
	jobDone       jobStatus = "done"
	jobFailed     jobStatus = "failed"
)

type Job struct {
	ID         string
	Request    ThumbnailRequest
	ImagePaths []string
	ImageFiles []imageFile
	JobDir     string
	RequestLog string
	OutputPath string
	JobURL     string
	ImageURL   string

	Status jobStatus
	Error  string

	CreatedAt  time.Time
	StartedAt  *time.Time
	FinishedAt *time.Time
	ExpiresAt  *time.Time

	Result  *ThumbnailResponse
	Webhook webhookDeliveryStatus
}

type JobSnapshot struct {
	ID            string
	Status        jobStatus
	Error         string
	QueuePosition *int
	PendingJobs   int
	CreatedAt     time.Time
	StartedAt     *time.Time
	FinishedAt    *time.Time
	ExpiresAt     *time.Time
	Result        *ThumbnailResponse
	OutputPath    string
	JobURL        string
	ImageURL      string
	Webhook       webhookDeliveryStatus
}

type JobManager struct {
	mu sync.RWMutex

	jobs         map[string]*Job
	queue        []string
	processingID string
	cond         *sync.Cond

	queueCapacity  int
	storageDir     string
	jobTTL         time.Duration
	jobTimeout     time.Duration
	webhookConfig  webhookConfig
	inlineMaxBytes int
	inlineMaxCount int
}

type apiServer struct {
	jobs      *JobManager
	startedAt time.Time
}

type webhookConfig struct {
	Timeout time.Duration
	Retries int
	Backoff time.Duration
}

type webhookDeliveryStatus struct {
	Enabled        bool       `json:"enabled"`
	URL            string     `json:"url,omitempty"`
	Attempts       int        `json:"attempts"`
	Delivered      bool       `json:"delivered"`
	LastStatusCode int        `json:"last_status_code,omitempty"`
	LastError      string     `json:"last_error,omitempty"`
	LastAttemptAt  *time.Time `json:"last_attempt_at,omitempty"`
	DeliveredAt    *time.Time `json:"delivered_at,omitempty"`
}

type enqueueResponse struct {
	JobID           string `json:"job_id"`
	Status          string `json:"status"`
	QueuePosition   int    `json:"queue_position"`
	PendingJobs     int    `json:"pending_jobs"`
	JobURL          string `json:"job_url"`
	ImageURL        string `json:"image_url"`
	CreatedAt       string `json:"created_at"`
	ExpiresInSecond int    `json:"expires_in_seconds"`
	WebhookEnabled  bool   `json:"webhook_enabled"`
}

type jobResponse struct {
	JobID         string                `json:"job_id"`
	Status        string                `json:"status"`
	QueuePosition *int                  `json:"queue_position,omitempty"`
	PendingJobs   int                   `json:"pending_jobs"`
	JobURL        string                `json:"job_url"`
	ImageURL      string                `json:"image_url"`
	CreatedAt     string                `json:"created_at"`
	StartedAt     string                `json:"started_at,omitempty"`
	FinishedAt    string                `json:"finished_at,omitempty"`
	ExpiresAt     string                `json:"expires_at,omitempty"`
	Result        *ThumbnailResponse    `json:"result,omitempty"`
	Error         string                `json:"error,omitempty"`
	Webhook       webhookDeliveryStatus `json:"webhook"`
}

type healthResponse struct {
	Status            string `json:"status"`
	Time              string `json:"time"`
	UptimeSeconds     int64  `json:"uptime_seconds"`
	QueuedJobs        int    `json:"queued_jobs"`
	HasProcessingJob  bool   `json:"has_processing_job"`
	TrackedJobs       int    `json:"tracked_jobs"`
	QueueCapacity     int    `json:"queue_capacity"`
	JobTTLSeconds     int64  `json:"job_ttl_seconds"`
	JobTimeoutSeconds int64  `json:"job_timeout_seconds"`
}

func main() {
	port := strings.TrimSpace(os.Getenv("PORT"))
	if port == "" {
		port = defaultPort
	}

	storageDir := strings.TrimSpace(os.Getenv("JOB_STORAGE_DIR"))
	if storageDir == "" {
		storageDir = "job-data"
	}

	queueCapacity := envInt("JOB_QUEUE_CAPACITY", defaultJobQueueCapacity)
	if queueCapacity < 1 {
		queueCapacity = defaultJobQueueCapacity
	}

	jobTTL := time.Duration(envInt("JOB_TTL_HOURS", defaultJobTTLHours)) * time.Hour
	if jobTTL < time.Minute {
		jobTTL = time.Duration(defaultJobTTLHours) * time.Hour
	}

	timeout := time.Duration(envInt("JOB_PROCESS_TIMEOUT_SECONDS", defaultJobProcessTimeoutSecs)) * time.Second
	if timeout < 30*time.Second {
		timeout = time.Duration(defaultJobProcessTimeoutSecs) * time.Second
	}

	webhookTimeout := time.Duration(envInt("WEBHOOK_TIMEOUT_SECONDS", defaultWebhookTimeoutSecs)) * time.Second
	if webhookTimeout < time.Second {
		webhookTimeout = time.Duration(defaultWebhookTimeoutSecs) * time.Second
	}
	webhookRetries := envInt("WEBHOOK_RETRIES", defaultWebhookRetries)
	if webhookRetries < 1 {
		webhookRetries = defaultWebhookRetries
	}
	webhookBackoff := time.Duration(envInt("WEBHOOK_BACKOFF_MS", defaultWebhookBackoffMs)) * time.Millisecond
	if webhookBackoff < 100*time.Millisecond {
		webhookBackoff = time.Duration(defaultWebhookBackoffMs) * time.Millisecond
	}
	inlineMaxBytes := envInt("INLINE_IMAGE_MAX_BYTES", defaultInlineImageMaxBytes)
	if inlineMaxBytes < (64 * 1024) {
		inlineMaxBytes = defaultInlineImageMaxBytes
	}
	inlineMaxCount := envInt("INLINE_IMAGE_MAX_COUNT", defaultInlineImageMaxCount)
	if inlineMaxCount < 1 {
		inlineMaxCount = defaultInlineImageMaxCount
	}

	manager, err := newJobManager(
		storageDir,
		queueCapacity,
		jobTTL,
		timeout,
		webhookConfig{
			Timeout: webhookTimeout,
			Retries: webhookRetries,
			Backoff: webhookBackoff,
		},
		inlineMaxBytes,
		inlineMaxCount,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "job manager init failed: %v\n", err)
		os.Exit(1)
	}
	manager.Start()

	serverImpl := &apiServer{
		jobs:      manager,
		startedAt: time.Now().UTC(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", serverImpl.healthHandler)
	mux.HandleFunc("/healthz", serverImpl.healthHandler)
	mux.HandleFunc("/thumbnail", serverImpl.thumbnailHandler)
	mux.HandleFunc("/job/", serverImpl.jobHandler)

	server := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	fmt.Printf("thumbnail API listening on :%s\n", port)
	fmt.Printf("job queue: capacity=%d, ttl=%s, storage=%s\n", queueCapacity, jobTTL, storageDir)
	fmt.Printf("webhook: timeout=%s retries=%d backoff=%s\n", webhookTimeout, webhookRetries, webhookBackoff)
	fmt.Printf("inline image: max_count=%d max_bytes=%d\n", inlineMaxCount, inlineMaxBytes)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintf(os.Stderr, "server error: %v\n", err)
		os.Exit(1)
	}
}

func newJobManager(storageDir string, queueCapacity int, ttl, timeout time.Duration, webhookCfg webhookConfig, inlineMaxBytes, inlineMaxCount int) (*JobManager, error) {
	absStorage, err := filepath.Abs(storageDir)
	if err != nil {
		return nil, fmt.Errorf("resolve job storage dir failed: %w", err)
	}
	if err := os.MkdirAll(absStorage, 0o755); err != nil {
		return nil, fmt.Errorf("create job storage dir failed: %w", err)
	}

	manager := &JobManager{
		jobs:           make(map[string]*Job),
		queue:          make([]string, 0),
		queueCapacity:  queueCapacity,
		storageDir:     absStorage,
		jobTTL:         ttl,
		jobTimeout:     timeout,
		webhookConfig:  webhookCfg,
		inlineMaxBytes: inlineMaxBytes,
		inlineMaxCount: inlineMaxCount,
	}
	manager.cond = sync.NewCond(&manager.mu)
	return manager, nil
}

func (m *JobManager) Start() {
	go m.workerLoop()
	go m.cleanupLoop()
}

func (m *JobManager) Enqueue(req ThumbnailRequest, imagePaths []string, imageFiles []imageFile, baseURL string) (*Job, int, int, error) {
	now := time.Now().UTC()
	jobID, err := newJobID()
	if err != nil {
		return nil, 0, 0, fmt.Errorf("generate job id failed: %w", err)
	}

	jobDir := filepath.Join(m.storageDir, jobID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		return nil, 0, 0, fmt.Errorf("create job dir failed: %w", err)
	}

	outputPath := filepath.Join(jobDir, "thumbnail.jpg")
	requestPath := filepath.Join(jobDir, "request.json")
	if err := writeJobPayloadLog(requestPath, req, imagePaths, outputPath, now); err != nil {
		_ = os.RemoveAll(jobDir)
		return nil, 0, 0, fmt.Errorf("write request payload failed: %w", err)
	}

	reqForJob := req
	reqForJob.ImageFiles = nil

	job := &Job{
		ID:         jobID,
		Request:    reqForJob,
		ImagePaths: append([]string(nil), imagePaths...),
		ImageFiles: append([]imageFile(nil), imageFiles...),
		JobDir:     jobDir,
		RequestLog: requestPath,
		OutputPath: outputPath,
		JobURL:     buildJobURLFromBase(baseURL, jobID),
		ImageURL:   buildJobImageURLFromBase(baseURL, jobID),
		Status:     jobQueued,
		CreatedAt:  now,
		Webhook: webhookDeliveryStatus{
			Enabled: strings.TrimSpace(req.WebhookURL) != "",
			URL:     strings.TrimSpace(req.WebhookURL),
		},
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.queue) >= m.queueCapacity {
		_ = os.RemoveAll(jobDir)
		return nil, 0, 0, errors.New("queue is full, please retry later")
	}

	m.jobs[job.ID] = job
	m.queue = append(m.queue, job.ID)
	queuePos := len(m.queue)
	pendingJobs := len(m.queue)
	if m.processingID != "" {
		pendingJobs++
	}

	m.cond.Signal()
	return job, queuePos, pendingJobs, nil
}

func (m *JobManager) workerLoop() {
	for {
		m.mu.Lock()
		for len(m.queue) == 0 {
			m.cond.Wait()
		}

		jobID := m.queue[0]
		m.queue = m.queue[1:]
		job, ok := m.jobs[jobID]
		if !ok {
			m.mu.Unlock()
			continue
		}

		startedAt := time.Now().UTC()
		job.Status = jobProcessing
		job.StartedAt = &startedAt
		m.processingID = jobID
		m.mu.Unlock()

		result, processErr := m.processJob(job)

		finishedAt := time.Now().UTC()
		expiresAt := finishedAt.Add(m.jobTTL)
		shouldDispatchWebhook := false

		m.mu.Lock()
		if current, exists := m.jobs[jobID]; exists {
			current.FinishedAt = &finishedAt
			current.ExpiresAt = &expiresAt
			if processErr != nil {
				current.Status = jobFailed
				current.Error = processErr.Error()
				current.Result = &ThumbnailResponse{
					Applied:         false,
					CompositionMode: "pair-smart-thumb",
					CropError:       processErr.Error(),
				}
			} else {
				current.Status = jobDone
				current.Error = ""
				current.Result = &result
			}
			shouldDispatchWebhook = current.Webhook.Enabled && strings.TrimSpace(current.Webhook.URL) != ""
		}
		m.processingID = ""
		m.mu.Unlock()

		if shouldDispatchWebhook {
			go m.deliverWebhook(jobID)
		}
	}
}

func (m *JobManager) processJob(job *Job) (ThumbnailResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), m.jobTimeout)
	defer cancel()

	allPaths := append([]string(nil), job.ImagePaths...)
	inlinePaths, err := m.writeInlineImageFiles(job)
	if err != nil {
		return ThumbnailResponse{}, err
	}
	allPaths = append(allPaths, inlinePaths...)
	if len(allPaths) == 0 {
		return ThumbnailResponse{}, errors.New("no valid input images found for job")
	}

	pair, err := runPythonSmartPair(ctx, allPaths, job.OutputPath)
	if err != nil {
		return ThumbnailResponse{}, err
	}

	resp := buildPairResponse(job.Request, pair, job.OutputPath)
	if !job.Request.ReturnCandidates {
		resp.Candidates = nil
	}
	return resp, nil
}

func (m *JobManager) writeInlineImageFiles(job *Job) ([]string, error) {
	if len(job.ImageFiles) == 0 {
		return nil, nil
	}
	if len(job.ImageFiles) > m.inlineMaxCount {
		return nil, fmt.Errorf("image_files exceeds max count (%d)", m.inlineMaxCount)
	}

	paths := make([]string, 0, len(job.ImageFiles))
	for idx, file := range job.ImageFiles {
		raw := strings.TrimSpace(file.ContentBase64)
		if raw == "" {
			return nil, fmt.Errorf("image_files[%d].content_base64 is required", idx)
		}

		bin, err := decodeBase64Image(raw)
		if err != nil {
			return nil, fmt.Errorf("image_files[%d] decode failed: %w", idx, err)
		}
		if len(bin) == 0 {
			return nil, fmt.Errorf("image_files[%d] is empty", idx)
		}
		if len(bin) > m.inlineMaxBytes {
			return nil, fmt.Errorf("image_files[%d] exceeds max bytes (%d)", idx, m.inlineMaxBytes)
		}

		filename := inlineImageFilename(file.Filename, idx)
		target := filepath.Join(job.JobDir, filename)
		if err := os.WriteFile(target, bin, 0o644); err != nil {
			return nil, fmt.Errorf("image_files[%d] write failed: %w", idx, err)
		}
		paths = append(paths, target)
	}
	return paths, nil
}

func decodeBase64Image(raw string) ([]byte, error) {
	text := strings.TrimSpace(raw)
	if strings.HasPrefix(strings.ToLower(text), "data:") {
		if cut := strings.Index(text, ","); cut >= 0 && cut < len(text)-1 {
			text = text[cut+1:]
		}
	}

	decoders := []func(string) ([]byte, error){
		base64.StdEncoding.DecodeString,
		base64.RawStdEncoding.DecodeString,
		base64.URLEncoding.DecodeString,
		base64.RawURLEncoding.DecodeString,
	}
	var lastErr error
	for _, decoder := range decoders {
		data, err := decoder(text)
		if err == nil {
			return data, nil
		}
		lastErr = err
	}
	if lastErr == nil {
		lastErr = errors.New("invalid base64")
	}
	return nil, lastErr
}

func inlineImageFilename(input string, idx int) string {
	ext := strings.ToLower(filepath.Ext(filepath.Base(strings.TrimSpace(input))))
	if ext == "" || len(ext) > 12 || strings.Contains(ext, "/") || strings.Contains(ext, "\\") {
		ext = ".img"
	}
	return fmt.Sprintf("upload-%03d%s", idx+1, ext)
}

func (m *JobManager) deliverWebhook(jobID string) {
	if m.webhookConfig.Retries < 1 {
		return
	}

	client := &http.Client{Timeout: m.webhookConfig.Timeout}

	for attempt := 1; attempt <= m.webhookConfig.Retries; attempt++ {
		webhookURL, payload, exists, payloadErr := m.buildWebhookPayload(jobID)
		if !exists {
			return
		}
		if strings.TrimSpace(webhookURL) == "" {
			return
		}
		if payloadErr != nil {
			m.recordWebhookAttempt(jobID, attempt, 0, payloadErr)
			return
		}

		statusCode, err := sendWebhook(client, webhookURL, payload)
		m.recordWebhookAttempt(jobID, attempt, statusCode, err)
		if err == nil && statusCode >= 200 && statusCode < 300 {
			return
		}

		if attempt < m.webhookConfig.Retries {
			time.Sleep(m.webhookConfig.Backoff)
		}
	}
}

func (m *JobManager) buildWebhookPayload(jobID string) (string, []byte, bool, error) {
	m.mu.RLock()
	job, ok := m.jobs[jobID]
	if !ok {
		m.mu.RUnlock()
		return "", nil, false, nil
	}

	webhookURL := strings.TrimSpace(job.Webhook.URL)
	payloadData := map[string]interface{}{
		"event":      "thumbnail.job.completed",
		"job_id":     job.ID,
		"status":     string(job.Status),
		"job_url":    job.JobURL,
		"image_url":  job.ImageURL,
		"created_at": job.CreatedAt.Format(time.RFC3339),
	}
	if job.StartedAt != nil {
		payloadData["started_at"] = job.StartedAt.Format(time.RFC3339)
	}
	if job.FinishedAt != nil {
		payloadData["finished_at"] = job.FinishedAt.Format(time.RFC3339)
	}
	if job.ExpiresAt != nil {
		payloadData["expires_at"] = job.ExpiresAt.Format(time.RFC3339)
	}
	if strings.TrimSpace(job.Error) != "" {
		payloadData["error"] = job.Error
	}
	if job.Result != nil {
		resultCopy := *job.Result
		if job.Status == jobDone {
			resultCopy.OutputPath = job.ImageURL
		}
		payloadData["result"] = resultCopy
	}
	m.mu.RUnlock()

	payload, err := json.Marshal(payloadData)
	if err != nil {
		return webhookURL, nil, true, fmt.Errorf("marshal webhook payload failed: %w", err)
	}
	return webhookURL, payload, true, nil
}

func sendWebhook(client *http.Client, webhookURL string, payload []byte) (int, error) {
	if len(payload) == 0 {
		return 0, errors.New("webhook payload is empty")
	}

	req, err := http.NewRequest(http.MethodPost, webhookURL, bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message := strings.TrimSpace(string(respBody))
		if message != "" {
			return resp.StatusCode, fmt.Errorf("non-2xx webhook response: %d %s", resp.StatusCode, truncateText(message, 300))
		}
		return resp.StatusCode, fmt.Errorf("non-2xx webhook response: %d", resp.StatusCode)
	}
	return resp.StatusCode, nil
}

func (m *JobManager) recordWebhookAttempt(jobID string, attempts int, statusCode int, attemptErr error) {
	now := time.Now().UTC()
	m.mu.Lock()
	defer m.mu.Unlock()

	job, ok := m.jobs[jobID]
	if !ok {
		return
	}

	job.Webhook.Attempts = attempts
	job.Webhook.LastStatusCode = statusCode
	job.Webhook.LastAttemptAt = &now
	if attemptErr != nil {
		job.Webhook.LastError = truncateText(attemptErr.Error(), 500)
		return
	}

	job.Webhook.Delivered = true
	job.Webhook.LastError = ""
	deliveredAt := now
	job.Webhook.DeliveredAt = &deliveredAt
}

func (m *JobManager) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		m.cleanupExpired(time.Now().UTC())
	}
}

func (m *JobManager) cleanupExpired(now time.Time) {
	expired := make([]*Job, 0)

	m.mu.Lock()
	for id, job := range m.jobs {
		if job.ExpiresAt == nil {
			continue
		}
		if now.Before(*job.ExpiresAt) {
			continue
		}
		delete(m.jobs, id)
		expired = append(expired, job)
	}
	m.mu.Unlock()

	for _, job := range expired {
		_ = os.RemoveAll(job.JobDir)
	}
}

func (m *JobManager) GetSnapshot(jobID string) (JobSnapshot, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	job, ok := m.jobs[jobID]
	if !ok {
		return JobSnapshot{}, false
	}

	snap := JobSnapshot{
		ID:          job.ID,
		Status:      job.Status,
		Error:       job.Error,
		PendingJobs: len(m.queue),
		CreatedAt:   job.CreatedAt,
		OutputPath:  job.OutputPath,
		JobURL:      job.JobURL,
		ImageURL:    job.ImageURL,
		Webhook:     cloneWebhookStatus(job.Webhook),
	}
	if m.processingID != "" {
		snap.PendingJobs++
	}

	if job.Status == jobQueued {
		pos := m.queuePositionLocked(job.ID)
		if pos > 0 {
			snap.QueuePosition = &pos
		}
	}
	if job.Status == jobProcessing {
		zero := 0
		snap.QueuePosition = &zero
	}

	if job.StartedAt != nil {
		started := *job.StartedAt
		snap.StartedAt = &started
	}
	if job.FinishedAt != nil {
		finished := *job.FinishedAt
		snap.FinishedAt = &finished
	}
	if job.ExpiresAt != nil {
		expires := *job.ExpiresAt
		snap.ExpiresAt = &expires
	}

	if job.Result != nil {
		resultCopy := *job.Result
		snap.Result = &resultCopy
	}

	return snap, true
}

func (m *JobManager) queuePositionLocked(jobID string) int {
	for idx, id := range m.queue {
		if id == jobID {
			return idx + 1
		}
	}
	return -1
}

func (m *JobManager) healthStats() (queuedJobs int, hasProcessing bool, trackedJobs int, queueCapacity int, ttlSeconds int64, timeoutSeconds int64) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	queuedJobs = len(m.queue)
	hasProcessing = m.processingID != ""
	trackedJobs = len(m.jobs)
	queueCapacity = m.queueCapacity
	ttlSeconds = int64(m.jobTTL.Seconds())
	timeoutSeconds = int64(m.jobTimeout.Seconds())
	return
}

func (s *apiServer) thumbnailHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		respondJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	var req ThumbnailRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}

	paths := collectInputPaths(req)
	imageFiles, err := sanitizeImageFiles(req.ImageFiles)
	if err != nil {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if len(paths) == 0 && len(imageFiles) == 0 {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": "image_paths, image_path, or image_files is required"})
		return
	}

	webhookURL, err := parseWebhookURL(req.WebhookURL)
	if err != nil {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	req.WebhookURL = webhookURL

	job, queuePos, pendingJobs, err := s.jobs.Enqueue(req, paths, imageFiles, buildBaseURL(r))
	if err != nil {
		respondJSON(w, http.StatusTooManyRequests, map[string]string{"error": err.Error()})
		return
	}

	resp := enqueueResponse{
		JobID:           job.ID,
		Status:          string(job.Status),
		QueuePosition:   queuePos,
		PendingJobs:     pendingJobs,
		JobURL:          job.JobURL,
		ImageURL:        job.ImageURL,
		CreatedAt:       job.CreatedAt.Format(time.RFC3339),
		ExpiresInSecond: int(s.jobs.jobTTL.Seconds()),
		WebhookEnabled:  strings.TrimSpace(job.Request.WebhookURL) != "",
	}
	respondJSON(w, http.StatusAccepted, resp)
}

func (s *apiServer) healthHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		respondJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	now := time.Now().UTC()
	queuedJobs, hasProcessing, trackedJobs, queueCapacity, ttlSeconds, timeoutSeconds := s.jobs.healthStats()
	resp := healthResponse{
		Status:            "ok",
		Time:              now.Format(time.RFC3339),
		UptimeSeconds:     int64(now.Sub(s.startedAt).Seconds()),
		QueuedJobs:        queuedJobs,
		HasProcessingJob:  hasProcessing,
		TrackedJobs:       trackedJobs,
		QueueCapacity:     queueCapacity,
		JobTTLSeconds:     ttlSeconds,
		JobTimeoutSeconds: timeoutSeconds,
	}
	respondJSON(w, http.StatusOK, resp)
}

func (s *apiServer) jobHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		respondJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/job/"), "/")
	if path == "" {
		respondJSON(w, http.StatusNotFound, map[string]string{"error": "job id is required"})
		return
	}

	parts := strings.Split(path, "/")
	jobID := strings.TrimSpace(parts[0])
	if jobID == "" {
		respondJSON(w, http.StatusNotFound, map[string]string{"error": "job id is required"})
		return
	}

	if len(parts) == 1 {
		s.jobStatusHandler(w, r, jobID)
		return
	}
	if len(parts) == 2 && parts[1] == "image" {
		s.jobImageHandler(w, r, jobID)
		return
	}

	respondJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
}

func (s *apiServer) jobStatusHandler(w http.ResponseWriter, r *http.Request, jobID string) {
	snap, ok := s.jobs.GetSnapshot(jobID)
	if !ok {
		respondJSON(w, http.StatusNotFound, map[string]string{"error": "job not found or expired"})
		return
	}

	jobURL := strings.TrimSpace(snap.JobURL)
	if jobURL == "" {
		jobURL = buildJobURL(r, jobID)
	}
	imageURL := strings.TrimSpace(snap.ImageURL)
	if imageURL == "" {
		imageURL = buildJobImageURL(r, jobID)
	}
	var result *ThumbnailResponse
	if snap.Result != nil {
		resultCopy := *snap.Result
		if snap.Status == jobDone {
			resultCopy.OutputPath = imageURL
		}
		result = &resultCopy
	}

	resp := jobResponse{
		JobID:         snap.ID,
		Status:        string(snap.Status),
		QueuePosition: snap.QueuePosition,
		PendingJobs:   snap.PendingJobs,
		JobURL:        jobURL,
		ImageURL:      imageURL,
		CreatedAt:     snap.CreatedAt.Format(time.RFC3339),
		Result:        result,
		Error:         snap.Error,
		Webhook:       snap.Webhook,
	}
	if snap.StartedAt != nil {
		resp.StartedAt = snap.StartedAt.Format(time.RFC3339)
	}
	if snap.FinishedAt != nil {
		resp.FinishedAt = snap.FinishedAt.Format(time.RFC3339)
	}
	if snap.ExpiresAt != nil {
		resp.ExpiresAt = snap.ExpiresAt.Format(time.RFC3339)
	}

	respondJSON(w, http.StatusOK, resp)
}

func (s *apiServer) jobImageHandler(w http.ResponseWriter, r *http.Request, jobID string) {
	snap, ok := s.jobs.GetSnapshot(jobID)
	if !ok {
		respondJSON(w, http.StatusNotFound, map[string]string{"error": "job not found or expired"})
		return
	}

	if snap.Status == jobQueued || snap.Status == jobProcessing {
		respondJSON(w, http.StatusConflict, map[string]string{"error": "job is not completed yet"})
		return
	}
	if snap.Status == jobFailed {
		respondJSON(w, http.StatusUnprocessableEntity, map[string]string{"error": "job failed, image is not available"})
		return
	}
	if _, err := os.Stat(snap.OutputPath); err != nil {
		respondJSON(w, http.StatusNotFound, map[string]string{"error": "output image was deleted or not found"})
		return
	}

	format, err := parseOutputFormat(r.URL.Query().Get("format"))
	if err != nil {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	width, err := parseOptionalInt(r.URL.Query().Get("width"), "width")
	if err != nil {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	quality, err := parseQuality(r.URL.Query().Get("quality"))
	if err != nil {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	if format == "jpg" && width == 0 && quality == defaultImageQuality {
		w.Header().Set("Content-Type", "image/jpeg")
		w.Header().Set("Cache-Control", "no-store")
		http.ServeFile(w, r, snap.OutputPath)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	data, contentType, err := runPythonImageConvert(ctx, snap.OutputPath, format, width, quality)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func collectInputPaths(req ThumbnailRequest) []string {
	paths := cleanImagePaths(req.ImagePaths)
	single := strings.TrimSpace(req.ImagePath)
	if single != "" {
		already := false
		for _, item := range paths {
			if filepath.Clean(item) == filepath.Clean(single) {
				already = true
				break
			}
		}
		if !already {
			paths = append(paths, single)
		}
	}
	return cleanImagePaths(paths)
}

func cleanImagePaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, path := range paths {
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err == nil {
			out = append(out, path)
		}
	}
	return out
}

func sanitizeImageFiles(files []imageFile) ([]imageFile, error) {
	if len(files) == 0 {
		return nil, nil
	}
	out := make([]imageFile, 0, len(files))
	for idx, file := range files {
		content := strings.TrimSpace(file.ContentBase64)
		if content == "" {
			return nil, fmt.Errorf("image_files[%d].content_base64 is required", idx)
		}
		out = append(out, imageFile{
			Filename:      strings.TrimSpace(file.Filename),
			ContentBase64: content,
		})
	}
	return out, nil
}

func runPythonSmartPair(ctx context.Context, imagePaths []string, outputPath string) (smartPairResult, error) {
	workerPath := strings.TrimSpace(os.Getenv("SMART_THUMB_WORKER_PATH"))
	if workerPath == "" {
		workerPath = "smart_thumb.py"
	}
	if !filepath.IsAbs(workerPath) {
		abs, err := filepath.Abs(workerPath)
		if err == nil {
			workerPath = abs
		}
	}
	if _, err := os.Stat(workerPath); err != nil {
		return smartPairResult{}, fmt.Errorf("smart pair worker not found at %s", workerPath)
	}

	payload := map[string]interface{}{
		"image_paths": imagePaths,
		"output_path": outputPath,
		"skip_edges":  envInt("THUMBNAIL_PAIR_SKIP_EDGES", 2),
		"gap":         envInt("THUMBNAIL_PAIR_GAP", 5),
		"width":       envInt("THUMBNAIL_PAIR_WIDTH", 1200),
	}
	if model := strings.TrimSpace(os.Getenv("THUMBNAIL_YOLO_MODEL")); model != "" {
		payload["yolo_model_name"] = model
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return smartPairResult{}, fmt.Errorf("marshal smart pair request failed: %w", err)
	}

	var attempts []string
	for _, cand := range pythonCandidates() {
		args := make([]string, 0, len(cand.Args)+3)
		args = append(args, cand.Args...)
		args = append(args, workerPath, "--input-json", string(body))

		cmd := exec.CommandContext(ctx, cand.Bin, args...)
		outBytes, cmdErr := cmd.CombinedOutput()
		output := strings.TrimSpace(string(outBytes))
		if cmdErr != nil {
			attempts = append(attempts, fmt.Sprintf("%s failed: %v", cand.Bin, cmdErr))
			continue
		}
		result, parseErr := parseSmartPairOutput(output)
		if parseErr != nil {
			attempts = append(attempts, fmt.Sprintf("%s output parse failed: %v", cand.Bin, parseErr))
			continue
		}
		if _, statErr := os.Stat(result.OutPath); statErr != nil {
			attempts = append(attempts, fmt.Sprintf("%s output file missing: %v", cand.Bin, statErr))
			continue
		}
		return result, nil
	}

	if len(attempts) == 0 {
		attempts = append(attempts, "no python candidate configured")
	}

	primaryErr := strings.Join(attempts, "; ")
	fallbackResult, fallbackErr := runPythonFallbackPair(ctx, imagePaths, outputPath, primaryErr)
	if fallbackErr == nil {
		fallbackResult.FallbackUsed = true
		if strings.TrimSpace(fallbackResult.FallbackReason) == "" {
			fallbackResult.FallbackReason = primaryErr
		}
		return fallbackResult, nil
	}
	return smartPairResult{}, fmt.Errorf("primary worker failed: %s; fallback failed: %v", primaryErr, fallbackErr)
}

func runPythonFallbackPair(ctx context.Context, imagePaths []string, outputPath, primaryError string) (smartPairResult, error) {
	workerPath := strings.TrimSpace(os.Getenv("SMART_THUMB_FALLBACK_WORKER_PATH"))
	if workerPath == "" {
		workerPath = "fallback_thumb.py"
	}
	if !filepath.IsAbs(workerPath) {
		abs, err := filepath.Abs(workerPath)
		if err == nil {
			workerPath = abs
		}
	}
	if _, err := os.Stat(workerPath); err != nil {
		return smartPairResult{}, fmt.Errorf("fallback worker not found at %s", workerPath)
	}

	payload := map[string]interface{}{
		"image_paths":      imagePaths,
		"output_path":      outputPath,
		"gap":              envInt("THUMBNAIL_PAIR_GAP", 5),
		"width":            envInt("THUMBNAIL_PAIR_WIDTH", 1200),
		"fallback_reason":  primaryError,
		"fallback_trigger": "primary-worker-failed",
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return smartPairResult{}, fmt.Errorf("marshal fallback request failed: %w", err)
	}

	var attempts []string
	for _, cand := range pythonCandidates() {
		args := make([]string, 0, len(cand.Args)+3)
		args = append(args, cand.Args...)
		args = append(args, workerPath, "--input-json", string(body))

		cmd := exec.CommandContext(ctx, cand.Bin, args...)
		outBytes, cmdErr := cmd.CombinedOutput()
		output := strings.TrimSpace(string(outBytes))
		if cmdErr != nil {
			attempts = append(attempts, fmt.Sprintf("%s failed: %v", cand.Bin, cmdErr))
			continue
		}

		result, parseErr := parseSmartPairOutput(output)
		if parseErr != nil {
			attempts = append(attempts, fmt.Sprintf("%s output parse failed: %v", cand.Bin, parseErr))
			continue
		}
		if _, statErr := os.Stat(result.OutPath); statErr != nil {
			attempts = append(attempts, fmt.Sprintf("%s output file missing: %v", cand.Bin, statErr))
			continue
		}
		result.FallbackUsed = true
		return result, nil
	}

	if len(attempts) == 0 {
		return smartPairResult{}, errors.New("no python candidate configured")
	}
	return smartPairResult{}, errors.New(strings.Join(attempts, "; "))
}

func runPythonImageConvert(ctx context.Context, inputPath, format string, width, quality int) ([]byte, string, error) {
	workerPath := strings.TrimSpace(os.Getenv("IMAGE_CONVERT_WORKER_PATH"))
	if workerPath == "" {
		workerPath = "image_convert.py"
	}
	if !filepath.IsAbs(workerPath) {
		abs, err := filepath.Abs(workerPath)
		if err == nil {
			workerPath = abs
		}
	}
	if _, err := os.Stat(workerPath); err != nil {
		return nil, "", fmt.Errorf("image convert worker not found at %s", workerPath)
	}

	baseArgs := []string{
		workerPath,
		"--input", inputPath,
		"--format", format,
		"--quality", strconv.Itoa(quality),
	}
	if width > 0 {
		baseArgs = append(baseArgs, "--width", strconv.Itoa(width))
	}

	var attempts []string
	for _, cand := range pythonCandidates() {
		args := make([]string, 0, len(cand.Args)+len(baseArgs))
		args = append(args, cand.Args...)
		args = append(args, baseArgs...)

		cmd := exec.CommandContext(ctx, cand.Bin, args...)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		outBytes, cmdErr := cmd.Output()
		if cmdErr != nil {
			msg := strings.TrimSpace(stderr.String())
			if msg == "" {
				msg = cmdErr.Error()
			}
			attempts = append(attempts, fmt.Sprintf("%s failed: %s", cand.Bin, truncateText(msg, 200)))
			continue
		}
		if len(outBytes) == 0 {
			attempts = append(attempts, fmt.Sprintf("%s failed: empty output", cand.Bin))
			continue
		}

		if format == "avif" {
			return outBytes, "image/avif", nil
		}
		return outBytes, "image/jpeg", nil
	}

	if len(attempts) == 0 {
		return nil, "", errors.New("no python candidate configured")
	}
	return nil, "", errors.New(strings.Join(attempts, "; "))
}

func parseSmartPairOutput(output string) (smartPairResult, error) {
	trimmed := strings.TrimSpace(output)
	if trimmed == "" {
		return smartPairResult{}, errors.New("empty smart pair output")
	}

	var result smartPairResult
	if err := json.Unmarshal([]byte(trimmed), &result); err == nil {
		if strings.TrimSpace(result.OutPath) == "" {
			return smartPairResult{}, errors.New("smart pair output missing out_path")
		}
		return result, nil
	}

	lines := strings.Split(trimmed, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if !strings.HasPrefix(line, "{") || !strings.HasSuffix(line, "}") {
			continue
		}
		if err := json.Unmarshal([]byte(line), &result); err == nil {
			if strings.TrimSpace(result.OutPath) == "" {
				return smartPairResult{}, errors.New("smart pair output missing out_path")
			}
			return result, nil
		}
	}
	return smartPairResult{}, errors.New("no JSON object found in smart pair output")
}

func buildPairResponse(req ThumbnailRequest, pair smartPairResult, fallbackOutputPath string) ThumbnailResponse {
	outPath := strings.TrimSpace(pair.OutPath)
	if outPath == "" {
		outPath = fallbackOutputPath
	}

	width := 1200
	height := 675
	if len(pair.Size) >= 2 {
		if pair.Size[0] > 0 {
			width = pair.Size[0]
		}
		if pair.Size[1] > 0 {
			height = pair.Size[1]
		}
	}

	resp := ThumbnailResponse{
		CropResult: CropResult{
			CropX:      0,
			CropY:      0,
			CropWidth:  width,
			CropHeight: height,
			Method:     "pair-smart-thumb",
			Confidence: 1.0,
		},
		Applied:         true,
		OutputPath:      outPath,
		CompositionMode: "pair-smart-thumb",
	}
	if pair.FallbackUsed {
		resp.FallbackUsed = true
		resp.WorkerWarning = "fallback-smart-thumb: lightweight portrait pair composer was used"
		if strings.TrimSpace(pair.FallbackReason) != "" {
			resp.WorkerWarning = resp.WorkerWarning + " (" + truncateText(pair.FallbackReason, 240) + ")"
		}
	}

	candidates := make([]ThumbnailCandidate, 0, len(pair.Picked))
	composedFrom := make([]string, 0, len(pair.Picked))
	for i, picked := range pair.Picked {
		if strings.TrimSpace(picked.Path) != "" {
			composedFrom = append(composedFrom, picked.Path)
		}

		candidate := ThumbnailCandidate{
			PageIndex:  picked.Idx,
			ImagePath:  picked.Path,
			Method:     "smart-thumb",
			Confidence: clamp(1.0-picked.TextRatio, 0.0, 1.0),
			Score:      picked.Total,
		}
		if len(picked.BBox) >= 4 {
			x0 := picked.BBox[0]
			y0 := picked.BBox[1]
			x1 := picked.BBox[2]
			y1 := picked.BBox[3]
			candidate.CropX = x0
			candidate.CropY = y0
			candidate.CropWidth = maxInt(1, x1-x0)
			candidate.CropHeight = maxInt(1, y1-y0)
		}
		candidates = append(candidates, candidate)

		if i == 0 {
			resp.SelectedImagePath = picked.Path
			resp.SelectedScore = picked.Total
			selectedIndex := picked.Idx
			resp.SelectedPageIndex = &selectedIndex
		}
	}
	resp.ComposedFrom = composedFrom
	if req.ReturnCandidates {
		resp.Candidates = candidates
	}
	return resp
}

func parseOutputFormat(raw string) (string, error) {
	format := strings.TrimSpace(strings.ToLower(raw))
	if format == "" {
		return "jpg", nil
	}
	switch format {
	case "jpg", "jpeg":
		return "jpg", nil
	case "avif":
		return "avif", nil
	default:
		return "", errors.New("format must be one of: jpg, avif")
	}
}

func parseOptionalInt(raw, name string) (int, error) {
	text := strings.TrimSpace(raw)
	if text == "" {
		return 0, nil
	}
	value, err := strconv.Atoi(text)
	if err != nil {
		return 0, fmt.Errorf("%s must be a positive integer", name)
	}
	if value <= 0 {
		return 0, fmt.Errorf("%s must be greater than 0", name)
	}
	return value, nil
}

func parseQuality(raw string) (int, error) {
	text := strings.TrimSpace(raw)
	if text == "" {
		return defaultImageQuality, nil
	}
	value, err := strconv.Atoi(text)
	if err != nil {
		return 0, errors.New("quality must be an integer between 0 and 100")
	}
	if value < 0 || value > 100 {
		return 0, errors.New("quality must be between 0 and 100")
	}
	return value, nil
}

func writeJobPayloadLog(path string, req ThumbnailRequest, imagePaths []string, outputPath string, createdAt time.Time) error {
	reqForLog := req
	inlineFileNames := make([]string, 0, len(req.ImageFiles))
	for _, file := range req.ImageFiles {
		inlineFileNames = append(inlineFileNames, strings.TrimSpace(file.Filename))
	}
	reqForLog.ImageFiles = nil

	logBody := map[string]interface{}{
		"created_at":           createdAt.Format(time.RFC3339),
		"request":              reqForLog,
		"resolved_image_paths": imagePaths,
		"resolved_output_path": outputPath,
		"inline_image_count":   len(req.ImageFiles),
		"inline_image_names":   inlineFileNames,
	}
	data, err := json.MarshalIndent(logBody, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func newJobID() (string, error) {
	buf := make([]byte, 12)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return hex.EncodeToString(buf), nil
}

func pythonCandidates() []commandCandidate {
	custom := strings.TrimSpace(os.Getenv("PYTHON_BIN"))
	candidates := make([]commandCandidate, 0, 3)
	if custom != "" {
		parts := strings.Fields(custom)
		if len(parts) > 0 {
			candidates = append(candidates, commandCandidate{
				Bin:  parts[0],
				Args: parts[1:],
			})
		}
	}
	candidates = append(candidates, commandCandidate{Bin: "python"})
	candidates = append(candidates, commandCandidate{Bin: "python3"})
	candidates = append(candidates, commandCandidate{Bin: "py", Args: []string{"-3"}})
	return candidates
}

func envInt(name string, defaultValue int) int {
	text := strings.TrimSpace(os.Getenv(name))
	if text == "" {
		return defaultValue
	}
	value, err := strconv.Atoi(text)
	if err != nil {
		return defaultValue
	}
	return value
}

func parseWebhookURL(raw string) (string, error) {
	webhookURL := strings.TrimSpace(raw)
	if webhookURL == "" {
		return "", nil
	}

	parsed, err := url.Parse(webhookURL)
	if err != nil {
		return "", errors.New("webhook_url is invalid")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", errors.New("webhook_url must use http or https")
	}
	if strings.TrimSpace(parsed.Host) == "" {
		return "", errors.New("webhook_url host is required")
	}
	return parsed.String(), nil
}

func buildBaseURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	if proto := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto")); proto != "" {
		scheme = strings.Split(proto, ",")[0]
	}
	return fmt.Sprintf("%s://%s", scheme, r.Host)
}

func buildJobURLFromBase(baseURL string, jobID string) string {
	return strings.TrimRight(strings.TrimSpace(baseURL), "/") + "/job/" + jobID
}

func buildJobImageURLFromBase(baseURL string, jobID string) string {
	return strings.TrimRight(strings.TrimSpace(baseURL), "/") + "/job/" + jobID + "/image"
}

func buildJobURL(r *http.Request, jobID string) string {
	return buildJobURLFromBase(buildBaseURL(r), jobID)
}

func buildJobImageURL(r *http.Request, jobID string) string {
	return buildJobImageURLFromBase(buildBaseURL(r), jobID)
}

func cloneWebhookStatus(in webhookDeliveryStatus) webhookDeliveryStatus {
	out := in
	if in.LastAttemptAt != nil {
		lastAttempt := *in.LastAttemptAt
		out.LastAttemptAt = &lastAttempt
	}
	if in.DeliveredAt != nil {
		deliveredAt := *in.DeliveredAt
		out.DeliveredAt = &deliveredAt
	}
	return out
}

func respondJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func clamp(v, low, high float64) float64 {
	if v < low {
		return low
	}
	if v > high {
		return high
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func truncateText(text string, maxLen int) string {
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen] + "..."
}
