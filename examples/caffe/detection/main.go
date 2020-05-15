package main

import (
	"context"
	"fmt"
	"image"
	"io/ioutil"
	"path/filepath"
	"sort"
	"math"
	"strings"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/k0kubun/pp"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	cupti "github.com/rai-project/go-cupti"
	"github.com/rai-project/go-tensorrt"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/all"
	"github.com/rai-project/tracer/ctimer"
	gotensor "gorgonia.org/tensor"
)

var (
	batchSize  = 1
	model      = "yolov3"
	shapeInput1 = []int{1, 3, 416, 416}
	shapeInput2 = []int{1, 2}
	mean       = []float32{0, 0, 0}
	scale      = []float32{255, 255, 255}
	baseDir, _ = filepath.Abs("../../../_fixtures")
	imgPath    = filepath.Join(baseDir, "lane_control.jpg")
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

func softmax(scores []float32) ([]float32) {
	var maxNum, expSum float32 = float32(math.MaxFloat32), 0.0
	probs := make([]float32, len(scores))
	for _, score := range scores {
		if score < maxNum {
			maxNum = score
		}
	}
	for i, score := range scores {
		exp := float32(math.Exp(float64(score - maxNum)))
		probs[i] = exp
		expSum += exp
	}
	for i, prob := range probs {
		probs[i] = prob / expSum
	}

	return probs
}

func main() {
	defer tracer.Close()

	dir := filepath.Join(baseDir, model)
	graph := filepath.Join(dir, model+".onnx")
	synset := filepath.Join(dir, "synset.txt")

	img, err := imgio.Open(imgPath)
	if err != nil {
		panic(err)
	}

	height := shapeInput1[2]
	width := shapeInput1[3]

	resized := transform.Resize(img, height, width, transform.Linear)
	input, err := cvtRGBImageToNCHW1DArray(resized, mean, scale)
	if err != nil {
		panic(err)
	}

	opts := options.New()

	if !nvidiasmi.HasGPU {
		panic("no GPU")
	}
	device := options.CUDA_DEVICE

	ctx := context.Background()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "tensorrt_onnx_yolov3")
	defer span.Finish()

	inputNodes := []options.Node{
		options.Node{
			Key:   "input1",
			Shape: shapeInput1,
			Dtype: gotensor.Float32,
		},
		options.Node{
			Key:   "image_shape",
			Shape: shapeInput1,
			Dtype: gotensor.Float32,
		},
	}

	outputNodes := []options.Node{
		options.Node{
			Key:   "yolonms_layer_1/ExpandDims_1:0",
			Dtype: gotensor.Float32,
		},
		options.Node{
			Key:   "yolonms_layer_1/ExpandDims_3:0",
			Dtype: gotensor.Float32,
		},
		options.Node{
			Key:   "yolonms_layer_1/concat_2:0",
			Dtype: gotensor.Float32,
		},
	}

	predictor, err := tensorrt.New(
		ctx,
		options.WithOptions(opts),
		options.Device(device, 0),
		options.Graph([]byte(graph)),
		options.BatchSize(batchSize),
		options.InputNodes(inputNodes),
		options.OutputNodes(outputNodes),
	)
	if err != nil {
		panic(fmt.Sprintf("%v", err))
	}
	defer predictor.Close()

	pp.Println("input size:", len(input), "byte_count:", len(input)*4)
	for ii := 0; ii < 3; ii++ {
		err = predictor.Predict(ctx, input)
		if err != nil {
			panic(err)
		}
	}

	enableCupti := true
	var cu *cupti.CUPTI
	if enableCupti {
		cu, err = cupti.New(cupti.Context(ctx))
		if err != nil {
			panic(err)
		}
	}

	predictor.StartProfiling("predict", "")

	err = predictor.Predict(ctx, input)
	if err != nil {
		panic(err)
	}

	predictor.EndProfiling()

	if enableCupti {
		cu.Wait()
		cu.Close()
	}

	profBuffer, err := predictor.ReadProfile()
	if err != nil {
		panic(err)
	}

	t, err := ctimer.New(profBuffer)
	if err != nil {
		panic(err)
	}
	t.Publish(ctx, tracer.FRAMEWORK_TRACE)

	outputs, err := predictor.ReadPredictionOutputs(ctx)
	if err != nil {
		panic(err)
	}

	pp.Println("Final output:", len(outputs))
	output := outputs[0]
	// output = softmax(output)
	labelsFileContent, err := ioutil.ReadFile(synset)
	if err != nil {
		panic(err)
	}
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

	results := features[0]
	for i := 0; i < 3; i++ {
		prediction := results[i]
		pp.Println(prediction.Probability, prediction.GetClassification().GetIndex(), prediction.GetClassification().GetLabel())
	}
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
