package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	defaultRatio           = 16.0 / 9.0
	defaultMaxAnalysisSize = 512
	defaultPort            = "8080"
)

type ThumbnailRequest struct {
	ImagePath        string      `json:"image_path"`
	ImagePaths       []string    `json:"image_paths"`
	ImageWidth       int         `json:"image_width"`
	ImageHeight      int         `json:"image_height"`
	PreferredRatio   interface{} `json:"preferred_ratio"`
	MaxAnalysisSize  int         `json:"max_analysis_size"`
	ApplyCrop        bool        `json:"apply_crop"`
	OutputPath       string      `json:"output_path"`
	Quality          int         `json:"quality"`
	ReturnCandidates bool        `json:"return_candidates"`
}

type CropResult struct {
	CropX      int     `json:"crop_x"`
	CropY      int     `json:"crop_y"`
	CropWidth  int     `json:"crop_width"`
	CropHeight int     `json:"crop_height"`
	Method     string  `json:"method"`
	Confidence float64 `json:"confidence"`
}

type ThumbnailResponse struct {
	CropResult
	Applied           bool                 `json:"applied"`
	OutputPath        string               `json:"output_path,omitempty"`
	WorkerWarning     string               `json:"worker_warning,omitempty"`
	CropError         string               `json:"crop_error,omitempty"`
	SelectedPageIndex *int                 `json:"selected_page_index,omitempty"`
	SelectedImagePath string               `json:"selected_image_path,omitempty"`
	SelectedScore     float64              `json:"selected_score,omitempty"`
	Candidates        []ThumbnailCandidate `json:"candidates,omitempty"`
}

type ThumbnailCandidate struct {
	PageIndex     int     `json:"page_index"`
	ImagePath     string  `json:"image_path"`
	CropX         int     `json:"crop_x"`
	CropY         int     `json:"crop_y"`
	CropWidth     int     `json:"crop_width"`
	CropHeight    int     `json:"crop_height"`
	Method        string  `json:"method"`
	Confidence    float64 `json:"confidence"`
	Score         float64 `json:"score"`
	WorkerWarning string  `json:"worker_warning,omitempty"`
}

type commandCandidate struct {
	Bin  string
	Args []string
}

func main() {
	port := os.Getenv("PORT")
	if strings.TrimSpace(port) == "" {
		port = defaultPort
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/thumbnail", thumbnailHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		respondJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	server := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	fmt.Printf("thumbnail API listening on :%s\n", port)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintf(os.Stderr, "server error: %v\n", err)
		os.Exit(1)
	}
}

func thumbnailHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		respondJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	var req ThumbnailRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&req); err != nil {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}

	resp := ThumbnailResponse{Applied: false}
	sourceImagePath := strings.TrimSpace(req.ImagePath)

	if len(cleanImagePaths(req.ImagePaths)) > 0 {
		resp = generateChapterThumbnail(r.Context(), req)
		sourceImagePath = resp.SelectedImagePath
	} else {
		crop, warnErr := GenerateThumbnailCrop(r.Context(), req)
		resp.CropResult = crop
		if warnErr != nil {
			resp.WorkerWarning = warnErr.Error()
		}
	}

	if shouldApplyCrop(req) {
		outputPath := resolveOutputPath(sourceImagePath, req.OutputPath)
		if err := applyCropWithVips(r.Context(), sourceImagePath, outputPath, resp.CropResult, req.Quality); err != nil {
			resp.CropError = err.Error()
		} else {
			resp.Applied = true
			resp.OutputPath = outputPath
		}
	}

	respondJSON(w, http.StatusOK, resp)
}

func generateChapterThumbnail(ctx context.Context, req ThumbnailRequest) ThumbnailResponse {
	paths := cleanImagePaths(req.ImagePaths)
	resp := ThumbnailResponse{Applied: false}
	if len(paths) == 0 {
		return resp
	}

	candidates := make([]ThumbnailCandidate, 0, len(paths))
	bestIndex := -1
	bestScore := -1.0
	warnings := make([]string, 0)

	for idx, imagePath := range paths {
		pageReq := req
		pageReq.ImagePath = imagePath
		pageReq.ImagePaths = nil
		pageReq.ImageWidth = 0
		pageReq.ImageHeight = 0

		crop, warnErr := GenerateThumbnailCrop(ctx, pageReq)
		score := scoreCandidate(crop)

		candidate := ThumbnailCandidate{
			PageIndex:  idx,
			ImagePath:  imagePath,
			CropX:      crop.CropX,
			CropY:      crop.CropY,
			CropWidth:  crop.CropWidth,
			CropHeight: crop.CropHeight,
			Method:     crop.Method,
			Confidence: crop.Confidence,
			Score:      roundFloat(score, 4),
		}

		if warnErr != nil {
			candidate.WorkerWarning = warnErr.Error()
			warnings = append(warnings, fmt.Sprintf("page %d: %s", idx, warnErr.Error()))
		}

		candidates = append(candidates, candidate)
		if isBetterCandidate(score, idx, bestScore, bestIndex) {
			bestScore = score
			bestIndex = idx
		}
	}

	if bestIndex < 0 {
		fallback := fallbackCrop(maxInt(req.ImageWidth, 0), maxInt(req.ImageHeight, 0), parseRatio(req.PreferredRatio))
		resp.CropResult = fallback
		resp.CropResult.Method = "fallback"
		resp.CropResult.Confidence = 0.0
	} else {
		selected := candidates[bestIndex]
		resp.CropResult = CropResult{
			CropX:      selected.CropX,
			CropY:      selected.CropY,
			CropWidth:  selected.CropWidth,
			CropHeight: selected.CropHeight,
			Method:     selected.Method,
			Confidence: selected.Confidence,
		}
		resp.SelectedImagePath = selected.ImagePath
		resp.SelectedScore = selected.Score
		selectedIndex := selected.PageIndex
		resp.SelectedPageIndex = &selectedIndex
	}

	if req.ReturnCandidates {
		resp.Candidates = candidates
	}
	if len(warnings) > 0 {
		resp.WorkerWarning = strings.Join(warnings, " | ")
	}

	return resp
}

func GenerateThumbnailCrop(ctx context.Context, req ThumbnailRequest) (CropResult, error) {
	ratio := parseRatio(req.PreferredRatio)
	if req.MaxAnalysisSize <= 0 {
		req.MaxAnalysisSize = defaultMaxAnalysisSize
	}
	if req.PreferredRatio == nil {
		req.PreferredRatio = "16:9"
	}

	width, height := resolveDimensions(req)

	result, err := runPythonWorker(ctx, req)
	if err == nil {
		bounded := ensureInBounds(result.CropX, result.CropY, result.CropWidth, result.CropHeight, width, height, ratio)
		result.CropX = bounded.CropX
		result.CropY = bounded.CropY
		result.CropWidth = bounded.CropWidth
		result.CropHeight = bounded.CropHeight
		if strings.TrimSpace(result.Method) == "" {
			result.Method = "fallback"
		}
		result.Confidence = clamp(result.Confidence, 0.0, 1.0)
		return result, nil
	}

	fallback := fallbackCrop(width, height, ratio)
	fallback.Method = "fallback"
	fallback.Confidence = 0.0
	return fallback, err
}

func runPythonWorker(ctx context.Context, req ThumbnailRequest) (CropResult, error) {
	workerPath := os.Getenv("THUMBNAIL_WORKER_PATH")
	if strings.TrimSpace(workerPath) == "" {
		workerPath = "thumbnail_worker.py"
	}
	if !filepath.IsAbs(workerPath) {
		abs, err := filepath.Abs(workerPath)
		if err == nil {
			workerPath = abs
		}
	}

	if _, err := os.Stat(workerPath); err != nil {
		return CropResult{}, fmt.Errorf("worker not found at %s", workerPath)
	}

	payload, err := json.Marshal(req)
	if err != nil {
		return CropResult{}, fmt.Errorf("marshal request failed: %w", err)
	}

	var attempts []string
	for _, cand := range pythonCandidates() {
		callArgs := make([]string, 0, len(cand.Args)+3)
		callArgs = append(callArgs, cand.Args...)
		callArgs = append(callArgs, workerPath, "--input-json", string(payload))

		cmd := exec.CommandContext(ctx, cand.Bin, callArgs...)
		cmdOutput, cmdErr := cmd.CombinedOutput()
		output := strings.TrimSpace(string(cmdOutput))
		if cmdErr != nil {
			attempts = append(attempts, fmt.Sprintf("%s failed: %v", cand.Bin, cmdErr))
			continue
		}

		parsed, parseErr := parseWorkerOutput(output)
		if parseErr != nil {
			attempts = append(attempts, fmt.Sprintf("%s output parse failed: %v", cand.Bin, parseErr))
			continue
		}
		return parsed, nil
	}

	if len(attempts) == 0 {
		return CropResult{}, errors.New("no python candidate configured")
	}
	return CropResult{}, errors.New(strings.Join(attempts, "; "))
}

func shouldApplyCrop(req ThumbnailRequest) bool {
	if req.ApplyCrop {
		return true
	}
	return strings.TrimSpace(req.OutputPath) != ""
}

func resolveOutputPath(inputPath, outputPath string) string {
	if strings.TrimSpace(outputPath) != "" {
		return outputPath
	}
	base := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	ext := strings.ToLower(filepath.Ext(inputPath))
	if ext == "" {
		ext = ".jpg"
	}
	return filepath.Join(filepath.Dir(inputPath), base+"_thumb"+ext)
}

func applyCropWithVips(ctx context.Context, inputPath, outputPath string, crop CropResult, quality int) error {
	if strings.TrimSpace(inputPath) == "" {
		return errors.New("image_path is required to apply crop")
	}
	if strings.TrimSpace(outputPath) == "" {
		return errors.New("output_path is empty")
	}
	if crop.CropWidth <= 0 || crop.CropHeight <= 0 {
		return errors.New("invalid crop size")
	}
	if _, err := os.Stat(inputPath); err != nil {
		return fmt.Errorf("image not found: %s", inputPath)
	}

	if quality <= 0 {
		quality = 85
	}
	quality = minInt(maxInt(quality, 1), 100)

	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return fmt.Errorf("create output directory failed: %w", err)
	}

	vipsBinary, err := exec.LookPath("vips")
	if err != nil {
		return errors.New("libvips CLI not found (install `vips` and ensure it is in PATH)")
	}

	outputSpec := buildVipsOutputSpec(outputPath, quality)
	args := []string{
		"crop",
		inputPath,
		outputSpec,
		strconv.Itoa(crop.CropX),
		strconv.Itoa(crop.CropY),
		strconv.Itoa(crop.CropWidth),
		strconv.Itoa(crop.CropHeight),
	}

	cmd := exec.CommandContext(ctx, vipsBinary, args...)
	out, cmdErr := cmd.CombinedOutput()
	if cmdErr != nil {
		stderr := strings.TrimSpace(string(out))
		if stderr != "" {
			return fmt.Errorf("vips crop failed: %v (%s)", cmdErr, stderr)
		}
		return fmt.Errorf("vips crop failed: %v", cmdErr)
	}
	return nil
}

func cleanImagePaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, path := range paths {
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}
		out = append(out, path)
	}
	return out
}

func scoreCandidate(crop CropResult) float64 {
	methodWeight := 0.5
	switch strings.ToLower(strings.TrimSpace(crop.Method)) {
	case "saliency":
		methodWeight = 0.9
	case "face":
		methodWeight = 1.0
	case "fallback":
		methodWeight = 0.55
	}
	score := (0.75 * clamp(crop.Confidence, 0.0, 1.0)) + (0.25 * methodWeight)
	return clamp(score, 0.0, 1.0)
}

func isBetterCandidate(score float64, pageIndex int, bestScore float64, bestPageIndex int) bool {
	const eps = 1e-9
	if bestPageIndex < 0 {
		return true
	}
	if score > bestScore+eps {
		return true
	}
	if math.Abs(score-bestScore) <= eps && pageIndex < bestPageIndex {
		return true
	}
	return false
}

func buildVipsOutputSpec(outputPath string, quality int) string {
	ext := strings.ToLower(filepath.Ext(outputPath))
	switch ext {
	case ".jpg", ".jpeg":
		return fmt.Sprintf("%s[Q=%d,optimize_coding,strip]", outputPath, quality)
	case ".webp":
		return fmt.Sprintf("%s[Q=%d,strip]", outputPath, quality)
	case ".png":
		return fmt.Sprintf("%s[compression=6,strip]", outputPath)
	default:
		return outputPath
	}
}

func parseWorkerOutput(output string) (CropResult, error) {
	trimmed := strings.TrimSpace(output)
	if trimmed == "" {
		return CropResult{}, errors.New("empty worker output")
	}

	var result CropResult
	if err := json.Unmarshal([]byte(trimmed), &result); err == nil {
		return result, nil
	}

	lines := strings.Split(trimmed, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if !strings.HasPrefix(line, "{") || !strings.HasSuffix(line, "}") {
			continue
		}
		if err := json.Unmarshal([]byte(line), &result); err == nil {
			return result, nil
		}
	}
	return CropResult{}, errors.New("no JSON object found in worker output")
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
	candidates = append(candidates, commandCandidate{Bin: "py", Args: []string{"-3"}})
	return candidates
}

func resolveDimensions(req ThumbnailRequest) (int, int) {
	w, h := req.ImageWidth, req.ImageHeight
	if w > 0 && h > 0 {
		return w, h
	}
	if strings.TrimSpace(req.ImagePath) == "" {
		return maxInt(w, 0), maxInt(h, 0)
	}

	file, err := os.Open(req.ImagePath)
	if err != nil {
		return maxInt(w, 0), maxInt(h, 0)
	}
	defer file.Close()

	cfg, _, err := image.DecodeConfig(file)
	if err != nil {
		return maxInt(w, 0), maxInt(h, 0)
	}

	if w <= 0 {
		w = cfg.Width
	}
	if h <= 0 {
		h = cfg.Height
	}
	return maxInt(w, 0), maxInt(h, 0)
}

func fallbackCrop(width, height int, ratio float64) CropResult {
	if width <= 0 || height <= 0 {
		return CropResult{
			CropX:      0,
			CropY:      0,
			CropWidth:  0,
			CropHeight: 0,
			Method:     "fallback",
			Confidence: 0.0,
		}
	}

	targetW := width
	targetH := int(math.Round(float64(targetW) / ratio))
	topBias := 0.18

	if targetH <= height {
		x := 0
		y := int(math.Round(float64(height) * topBias))
		if y+targetH > height {
			y = maxInt(0, height-targetH)
		}
		return CropResult{
			CropX:      x,
			CropY:      y,
			CropWidth:  targetW,
			CropHeight: targetH,
			Method:     "fallback",
			Confidence: 0.0,
		}
	}

	targetH = height
	targetW = int(math.Round(float64(targetH) * ratio))
	if targetW > width {
		targetW = width
		targetH = int(math.Round(float64(targetW) / ratio))
		if targetH > height {
			targetH = height
		}
	}

	x := maxInt(0, (width-targetW)/2)
	y := 0
	return CropResult{
		CropX:      x,
		CropY:      y,
		CropWidth:  targetW,
		CropHeight: targetH,
		Method:     "fallback",
		Confidence: 0.0,
	}
}

func ensureInBounds(x, y, w, h, imageW, imageH int, ratio float64) CropResult {
	if imageW <= 0 || imageH <= 0 {
		return CropResult{CropX: 0, CropY: 0, CropWidth: maxInt(w, 0), CropHeight: maxInt(h, 0)}
	}

	if w <= 0 || h <= 0 {
		return fallbackCrop(imageW, imageH, ratio)
	}

	w = minInt(maxInt(w, 1), imageW)
	h = minInt(maxInt(h, 1), imageH)
	w, h = fitRatioSize(float64(w), float64(h), imageW, imageH, ratio)

	x = minInt(maxInt(x, 0), imageW-w)
	y = minInt(maxInt(y, 0), imageH-h)

	return CropResult{
		CropX:      x,
		CropY:      y,
		CropWidth:  w,
		CropHeight: h,
	}
}

func fitRatioSize(desiredW, desiredH float64, imageW, imageH int, ratio float64) (int, int) {
	w := math.Max(1.0, desiredW)
	h := math.Max(1.0, desiredH)

	if w/h > ratio {
		w = h * ratio
	} else {
		h = w / ratio
	}

	if w > float64(imageW) {
		w = float64(imageW)
		h = w / ratio
	}
	if h > float64(imageH) {
		h = float64(imageH)
		w = h * ratio
	}

	wi := minInt(maxInt(int(math.Round(w)), 1), imageW)
	hi := minInt(maxInt(int(math.Round(h)), 1), imageH)

	if float64(wi)/float64(hi) > ratio {
		wi = minInt(maxInt(int(math.Round(float64(hi)*ratio)), 1), imageW)
	} else {
		hi = minInt(maxInt(int(math.Round(float64(wi)/ratio)), 1), imageH)
	}

	return wi, hi
}

func parseRatio(value interface{}) float64 {
	if value == nil {
		return defaultRatio
	}

	switch v := value.(type) {
	case float64:
		if v > 0 {
			return v
		}
		return defaultRatio
	case float32:
		if v > 0 {
			return float64(v)
		}
		return defaultRatio
	case int:
		if v > 0 {
			return float64(v)
		}
		return defaultRatio
	case json.Number:
		n, err := v.Float64()
		if err == nil && n > 0 {
			return n
		}
		return defaultRatio
	case string:
		text := strings.TrimSpace(v)
		if text == "" {
			return defaultRatio
		}
		if strings.Contains(text, ":") {
			parts := strings.SplitN(text, ":", 2)
			if len(parts) == 2 {
				left, lErr := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
				right, rErr := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
				if lErr == nil && rErr == nil && left > 0 && right > 0 {
					return left / right
				}
			}
		}
		num, err := strconv.ParseFloat(text, 64)
		if err == nil && num > 0 {
			return num
		}
		return defaultRatio
	default:
		return defaultRatio
	}
}

func respondJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func clamp(v, low, high float64) float64 {
	return math.Max(low, math.Min(high, v))
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func roundFloat(v float64, digits int) float64 {
	if digits < 0 {
		return v
	}
	factor := math.Pow(10, float64(digits))
	return math.Round(v*factor) / factor
}
