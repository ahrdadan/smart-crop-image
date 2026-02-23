package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const defaultPort = "8080"

type ThumbnailRequest struct {
	ImagePath        string      `json:"image_path"`
	ImagePaths       []string    `json:"image_paths"`
	OutputPath       string      `json:"output_path"`
	ReturnCandidates bool        `json:"return_candidates"`
	ComposeMode      string      `json:"compose_mode"`
	PreferredRatio   interface{} `json:"preferred_ratio"`
	MaxAnalysisSize  int         `json:"max_analysis_size"`
	ApplyCrop        bool        `json:"apply_crop"`
	Quality          int         `json:"quality"`
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
	OutPath string            `json:"out_path"`
	Size    []int             `json:"size"`
	Picked  []smartPairPicked `json:"picked"`
}

type commandCandidate struct {
	Bin  string
	Args []string
}

func main() {
	port := strings.TrimSpace(os.Getenv("PORT"))
	if port == "" {
		port = defaultPort
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/thumbnail", thumbnailHandler)

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
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}

	paths := collectInputPaths(req)
	if len(paths) == 0 {
		respondJSON(w, http.StatusBadRequest, map[string]string{"error": "image_paths or image_path is required"})
		return
	}

	outputPath := resolveOutputPath(paths[0], req.OutputPath)
	resp := ThumbnailResponse{
		Applied:         false,
		OutputPath:      outputPath,
		CompositionMode: "pair-smart-thumb",
	}

	pair, err := runPythonSmartPair(r.Context(), paths, outputPath)
	if err != nil {
		resp.CropError = err.Error()
		respondJSON(w, http.StatusOK, resp)
		return
	}

	resp = buildPairResponse(req, pair, outputPath)
	if !req.ReturnCandidates {
		resp.Candidates = nil
	}
	respondJSON(w, http.StatusOK, resp)
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
		return smartPairResult{}, errors.New("no python candidate configured")
	}
	return smartPairResult{}, errors.New(strings.Join(attempts, "; "))
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

func resolveOutputPath(inputPath, outputPath string) string {
	if strings.TrimSpace(outputPath) != "" {
		return outputPath
	}
	base := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	return filepath.Join(filepath.Dir(inputPath), base+"_thumb.jpg")
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
