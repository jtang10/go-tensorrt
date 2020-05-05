// +build linux
// +build !ppc64le
// +build !nogpu
// +build cgo

package tensorrt

import (
	"context"
<<<<<<< HEAD
=======
	"fmt"
	"image"
>>>>>>> rai_master
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/GeertJohan/go-sourcepath"
<<<<<<< HEAD
=======
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
>>>>>>> rai_master
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
<<<<<<< HEAD
	"github.com/rai-project/image"
	"github.com/rai-project/image/types"
	"github.com/stretchr/testify/assert"
)

var (
	batchSize       = 1
	thisDir         = sourcepath.MustAbsoluteDir()
	labelFilePath   = filepath.Join(thisDir, "_fixtures", "networks", "ilsvrc12_synset_words.txt")
	graphFilePath   = filepath.Join(thisDir, "_fixtures", "networks", "googlenet.prototxt")
	weightsFilePath = filepath.Join(thisDir, "_fixtures", "networks", "bvlc_googlenet.caffemodel")
)

func TestTensorRT(t *testing.T) {

	reader, _ := os.Open(filepath.Join(thisDir, "_fixtures", "cat.jpg"))
	defer reader.Close()

	img0, err := image.Read(reader, image.Resized(224, 224))
	assert.NoError(t, err)

	img := img0.(*types.RGBImage)

	const channels = 3
	bounds := img.Bounds()
	w, h := bounds.Max.X, bounds.Max.Y
	imgArray := make([]float32, w*h*3)
	pixels := img.Pix

	mean := []float32{104.0069879317889, 116.66876761696767, 122.6789143406786}
	scale := float32(1.0)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			width, height := w, h
			offset := y*img.Stride + x*3
			rgb := pixels[offset : offset+3]
			r, g, b := rgb[0], rgb[1], rgb[2]
			imgArray[y*width+x] = (float32(b) - mean[0]) / scale
			imgArray[width*height+y*width+x] = (float32(g) - mean[1]) / scale
			imgArray[2*width*height+y*width+x] = (float32(r) - mean[2]) / scale

		}
	}

	ctx := context.Background()
	predictor, err := New(
		ctx,
		options.Graph([]byte(graphFilePath)),
		options.Weights([]byte(weightsFilePath)),
		options.BatchSize(1),
		options.InputNode("data", []int{3, 224, 224}),
		options.OutputNode("prob"),
	)
	if err != nil {
		t.Errorf("tensorRT initiate failed %v", err)
=======
	nvidiasmi "github.com/rai-project/nvidia-smi"
	_ "github.com/rai-project/tracer/all"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

var (
	batchSize            = 1
	shape                = []int{1, 3, 224, 224}
	mean                 = []float32{123.68, 116.779, 103.939}
	scale                = []float32{1.0, 1.0, 1.0}
	thisDir              = sourcepath.MustAbsoluteDir()
	imgPath              = filepath.Join(thisDir, "examples", "_fixtures", "platypus.jpg")
	labelFilePath        = filepath.Join(thisDir, "examples", "_fixtures", "resnet50", "synset.txt")
	caffeGraphFilePath   = filepath.Join(thisDir, "examples", "_fixtures", "resnet50", "resnet50.prototxt")
	caffeWeightsFilePath = filepath.Join(thisDir, "examples", "_fixtures", "resnet50", "resnet50.caffemodel")
	onnxModelPath        = filepath.Join(thisDir, "examples", "_fixtures", "ResNet50.onnx")
	uffModelPath         = filepath.Join(thisDir, "examples", "_fixtures", "resnet50-infer-5.uff")
)

// convert go RGB Image to 1D normalized RGB array
func cvtRGBImageToNCHW1DArray(src image.Image, mean []float32, scale []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	in := src.Bounds()
	height := in.Max.Y - in.Min.Y // image height
	width := in.Max.X - in.Min.X  // image width
	stride := width * height      // image size per channel

	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := src.At(x+in.Min.X, y+in.Min.Y).RGBA()
			out[0*stride+y*width+x] = (float32(r>>8) - mean[0]) / scale[0]
			out[1*stride+y*width+x] = (float32(g>>8) - mean[1]) / scale[1]
			out[2*stride+y*width+x] = (float32(b>>8) - mean[2]) / scale[2]
		}
	}

	return out, nil
}

func TestTensorRTCaffe(t *testing.T) {
	img, err := imgio.Open(imgPath)
	if err != nil {
		t.Errorf("Test input image is not found: %v", err)
	}

	height := shape[2]
	width := shape[3]

	var input []float32
	for ii := 0; ii < batchSize; ii++ {
		resized := transform.Resize(img, height, width, transform.Linear)
		res, err := cvtRGBImageToNCHW1DArray(resized, mean, scale)
		if err != nil {
			t.Errorf("Test input image transformation is not successful: %v", err)
		}
		input = append(input, res...)
	}

	opts := options.New()

	if !nvidiasmi.HasGPU {
		t.Errorf("GPU is not detected: %v", err)
	}
	device := options.CUDA_DEVICE

	ctx := context.Background()
	in := options.Node{
		Key:   "data",
		Shape: shape,
		Dtype: gotensor.Float32,
	}
	out := options.Node{
		Key:   "prob",
		Dtype: gotensor.Float32,
	}

	predictor, err := New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(caffeGraphFilePath)),
		options.Weights([]byte(caffeWeightsFilePath)),
		options.BatchSize(batchSize),
		options.InputNodes([]options.Node{in}),
		options.OutputNodes([]options.Node{out}),
	)
	if err != nil {
		t.Errorf("TensorRT predictor initiation failed %v", err)
>>>>>>> rai_master
	}

	defer predictor.Close()

<<<<<<< HEAD
	err = predictor.Predict(ctx, imgArray)
=======
	err = predictor.Predict(ctx, input)
>>>>>>> rai_master
	if err != nil {
		t.Errorf("tensorRT inference failed %v", err)
	}

<<<<<<< HEAD
	output, err := predictor.ReadPredictionOutput(ctx)
=======
	outputs, err := predictor.ReadPredictionOutputs(ctx)
>>>>>>> rai_master
	if err != nil {
		panic(err)
	}

<<<<<<< HEAD
=======
	output := outputs[0]
>>>>>>> rai_master
	labelsFileContent, err := ioutil.ReadFile(labelFilePath)
	assert.NoError(t, err)
	labels := strings.Split(string(labelsFileContent), "\n")

	features := make([]dlframework.Features, batchSize)
	featuresLen := len(output) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(output[ii*featuresLen+jj]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		features[ii] = rprobs
	}

	top1 := features[0][0]

<<<<<<< HEAD
	assert.Equal(t, int32(281), top1.GetClassification().GetIndex())

	if top1.GetClassification().GetLabel() != "n02123045 tabby, tabby cat" {
		t.Errorf("tensorRT class label wrong")
	}
	if math.Abs(float64(top1.GetProbability()-0.324)) > .001 {
=======
	assert.Equal(t, int32(103), top1.GetClassification().GetIndex())
	pp.Println(top1.GetClassification().GetLabel(), top1.GetProbability())
	if top1.GetClassification().GetLabel() != "n01873310 platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus" {
		t.Errorf("tensorRT class label wrong")
	}
	if math.Abs(float64(top1.GetProbability()-0.99)) > .01 {
>>>>>>> rai_master
		t.Errorf("tensorRT class probablity wrong")
	}
}

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)

	os.Exit(m.Run())
}
