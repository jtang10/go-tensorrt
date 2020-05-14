package tensorrt

import (
	"strings"
)

// ModelFormat ...
type ModelFormat int

const (
	ModelFormatCaffe            ModelFormat = 1
	ModelFormatOnnx             ModelFormat = 2
	ModelFormatSerializedEngine ModelFormat = 3
	ModelFormatUnknown          ModelFormat = 999
	// ModelFormatUff     ModelFormat = 3
)

func ClassifyModelFormat(path string) ModelFormat {
	var format ModelFormat
	if strings.HasSuffix(path, "prototxt") {
		format = ModelFormatCaffe
	} else if strings.HasSuffix(path, "onnx") {
		format = ModelFormatOnnx
	} else if strings.HasSuffix(path, "engine") {
		format = ModelFormatSerializedEngine
	} else {
		format = ModelFormatUnknown
	}

	return format
}
