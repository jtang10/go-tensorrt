#ifdef __linux__
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

#include <cuda_runtime_api.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "json.hpp"
#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"
#include "half.hpp"

// #define DEBUG true

using namespace nvinfer1;
using json = nlohmann::json;

static bool has_error = false;
static std::string error_string{""};

static void clear_error() {
  has_error = false;
  error_string = "";
}

static void set_error(const std::string &err) {
  has_error = true;
  error_string = err;
}

#define START_C_DEFINION()                                                     \
  clear_error();                                                               \
  try {

#define END_C_DEFINION(res)                                                    \
  }                                                                            \
  catch (const std::exception &e) {                                            \
    std::cerr << "ERROR: " << e.what() << "\n";                                \
    set_error(e.what());                                                       \
  }                                                                            \
  catch (const std::string &e) {                                               \
    std::cerr << "ERROR: " << e << "\n";                                       \
    set_error(e);                                                              \
  }                                                                            \
  catch (...) {                                                                \
    std::cerr << "ERROR: unknown exception in go-tensorrt"                     \
              << "\n";                                                         \
    set_error("unknown exception in go-tensorrt");                             \
  }                                                                            \
  clear_error();                                                               \
  return res

auto reportSeverity = ILogger::Severity::kWARNING;
class Logger : public ILogger {
  void log(Severity severity, const char *msg) override {
    // suppress info-level messages
    if (severity < reportSeverity) {
      std::cout << msg << std::endl;
    }
  }
} gLogger;

#define CHECK(stmt) stmt

#define CHECK_ERROR(stmt) stmt

class Profiler : public IProfiler {
public:
  Profiler(profile *prof) : prof_(prof) {
    if (prof_ == nullptr) {
      return;
    }
    prof_->start(); // reset start time
    current_time_ = prof_->get_start();
  }

  /** \brief layer time reporting callback
   *
   * \param layerName the name of the layer, set when constructing the network
   * definition
   * \param ms the time in milliseconds to execute the layer
   */
  virtual void reportLayerTime(const char *layer_name, float ms) {

    if (prof_ == nullptr) {
      return;
    }

    shapes_t shapes{};

    auto duration = std::chrono::nanoseconds((timestamp_t::rep)(1000000 * ms));
    auto e = new profile_entry(current_layer_sequence_index_, layer_name, "",
                               shapes);
    e->set_start(current_time_);
    e->set_end(current_time_ + duration);
    prof_->add(current_layer_sequence_index_ - 1, e);

    current_layer_sequence_index_++;
    current_time_ += duration;
  }

  virtual ~Profiler() {}

private:
  profile *prof_{nullptr};
  int current_layer_sequence_index_{1};
  timestamp_t current_time_{};
};

class Predictor {
public:
  Predictor(IExecutionContext *context,
            std::vector<std::string> input_layer_names,
            std::vector<std::string> output_layer_names, int32_t batch_size)
      : context_(context), input_layer_names_(input_layer_names),
        output_layer_names_(output_layer_names), batch_size_(batch_size) {
    cudaStreamCreate(&stream_);
    const ICudaEngine &engine = context_->getEngine();
    data_.resize(engine.getNbBindings());
  };
  void Run() {
    if (context_ == nullptr) {
      throw std::runtime_error("tensorrt prediction error  null context_");
    }
    const ICudaEngine &engine = context_->getEngine();

    if (engine.getNbBindings() !=
        input_layer_names_.size() + output_layer_names_.size()) {
      throw std::runtime_error(std::string("tensorrt prediction error on ") +
                               std::to_string(__LINE__));
    }

    Profiler profiler(prof_);

    // Set the custom profiler.
    context_->setProfiler(&profiler);

    if (engine.hasImplicitBatchDimension()) {
      context_->execute(batch_size_, data_.data());
    } else {
      context_->executeV2(data_.data());
    }
    // context_->enqueue(batch_size_, data_.data(), stream_, nullptr);
  }

  template <typename T>
  void AddInput(const std::string &name, const T *host_data,
                size_t num_elements) {
    void *gpu_data = nullptr;
    const ICudaEngine &engine = context_->getEngine();
    const auto idx = engine.getBindingIndex(name.c_str());
#ifdef DEBUG
    std::cout << "========== C AddInput ==========" << std::endl;
    std::cout << __LINE__ << "  >>> " << "Found " << name << " as input with index " << idx << " with dimension ";
    auto dims = engine.getBindingDimensions(idx);
    for (int i = 0; i < dims.nbDims; ++i) {
      std:: cout << dims.d[i] << " ";
    }
    std::cout << std::endl;
    std::cout << __LINE__ << "  >>> " << "num_elements: " << num_elements << std::endl;
#endif
    if (idx == -1) {
      throw std::runtime_error(std::string("invalid input name ") + name);
    }
    const auto byte_count = num_elements * sizeof(T);
#ifdef DEBUG
    std::cout << __LINE__ << "  >>> " << "batch_size_: " << batch_size_ << " byte_count: " << byte_count << std::endl << std::endl;
#endif
    CHECK_ERROR(cudaMalloc(&gpu_data, byte_count));
    CHECK_ERROR(cudaMemcpyAsync(gpu_data, host_data, byte_count,
                                cudaMemcpyHostToDevice, stream_));
    data_[idx] = gpu_data;
  }

  template <typename T> void AddOutput(const std::string &name) {
    void *gpu_data = nullptr;
    const ICudaEngine &engine = context_->getEngine();
    const auto idx = engine.getBindingIndex(name.c_str());
#ifdef DEBUG
    std::cout << "Found " << name << " as output with index " << idx << std::endl;
#endif
    if (idx == -1) {
      throw std::runtime_error(std::string("invalid output name ") + name);
    }
    const auto dims = engine.getBindingDimensions(idx);
    const auto ndims = dims.nbDims;
    auto num_elements = 1;
    std::vector<int> res{};
    for (int ii = 0; ii < ndims; ii++) {
      num_elements *= dims.d[ii];
    }
    const auto byte_count = batch_size_ * num_elements * sizeof(T);
#ifdef DEBUG
    std::cout << "========== C AddOutput ==========" << std::endl;
    std::cout << __LINE__ << "  >>> " << "output byte_count: " << byte_count << std::endl;
#endif
    CHECK_ERROR(cudaMalloc(&gpu_data, byte_count));
    data_[idx] = gpu_data;
  }

  void *GetOutputData(const std::string &name) {
    synchronize();

    const ICudaEngine &engine = context_->getEngine();
    const auto idx = engine.getBindingIndex(name.c_str());
    if (idx == -1) {
      throw std::runtime_error(std::string("invalid output name ") + name);
    }

    if (engine.bindingIsInput(idx)) {
      throw std::runtime_error(std::string("the layer name is not an output ") +
                               name);
    }

    const auto shape = GetOutputShape(name);
    auto element_byte_count = 1;
    const auto data_type = engine.getBindingDataType(idx);
    const size_t num_elements =
        std::accumulate(begin(shape), end(shape), 1, std::multiplies<size_t>());

    switch (data_type) {
#define DISPATCH_GET_OUTPUT(DType, CType)                                      \
    case DType:                                                                \
      element_byte_count = sizeof(CType);                                      \
      break;                                                                   \
      TensorRT_DType_Dispatch(DISPATCH_GET_OUTPUT)
#undef DISPATCH_GET_OUTPUT
    case DataType::kFLOAT:
      element_byte_count = sizeof(float);
      break;
    case DataType::kHALF:
      element_byte_count = sizeof(short);
      break;
    case DataType::kINT8:
      element_byte_count = sizeof(int8_t);
      break;
    case DataType::kINT32:
      element_byte_count = sizeof(int32_t);
      break;
    default:
      throw std::runtime_error("unexpected output type");
    }
    const auto byte_count = batch_size_ * num_elements * element_byte_count;
    void *res_data = malloc(byte_count);

#ifdef DEBUG
    std::cout << "========== GetOutputData ==========" << std::endl;
    std::cout << __LINE__ << "  >>> " << "shape = " << shape[0] << "\n";
    std::cout << __LINE__ << "  >>> " << "byte_count = " << byte_count << "\n";
#endif

    CHECK(cudaMemcpy(res_data, data_[idx], byte_count, cudaMemcpyDeviceToHost));
    return res_data;
  }

  std::vector<int32_t> GetOutputShape(const std::string &name) {
    synchronize();

    const ICudaEngine &engine = context_->getEngine();
    const auto idx = engine.getBindingIndex(name.c_str());
    if (idx == -1) {
      throw std::runtime_error(std::string("invalid output name ") + name);
    }

    const auto dims = engine.getBindingDimensions(idx);
    const auto ndims = dims.nbDims;
#ifdef DEBUG
    std::cout << "========== GetOutputShape ==========" << std::endl;
    std::cout << "name = " << name << "; ";
    std::cout << "ndims = " << ndims << "; ";
#endif
    std::vector<int> res{};
    for (int ii = 0; ii < ndims; ii++) {
#ifdef DEBUG
      std::cout << dims.d[ii] << " ";
#endif
      res.emplace_back(dims.d[ii]);
    }
#ifdef DEBUG
    std::cout << std::endl << __LINE__ << "  >>> "
              << "res.size() = " << res.size() << "\n";
#endif
    return res;
  }

  void synchronize() { CHECK(cudaStreamSynchronize(stream_)); }
  ~Predictor() {
    for (auto data : data_) {
      cudaFree(data);
    }
    if (context_) {
      context_->destroy();
    }
    if (prof_) {
      prof_->reset();
      delete prof_;
      prof_ = nullptr;
    }
  }

  IExecutionContext *context_{nullptr};
  std::vector<std::string> input_layer_names_{nullptr};
  std::vector<std::string> output_layer_names_{nullptr};
  int32_t batch_size_{1};
  std::vector<void *> data_{nullptr};
  cudaStream_t stream_{0};
  profile *prof_{nullptr};
  bool profile_enabled_{false};
};

Predictor *get_predictor_from_handle(PredictorHandle predictor_handle) {
  auto predictor = (Predictor *)predictor_handle;
  if (predictor == nullptr) {
    throw std::runtime_error("expecting a non-nil predictor");
  }
  return predictor;
}

DataType get_blob_data_type(TensorRT_DType model_datatype) {
  DataType blob_data_type = DataType::kFLOAT;
  switch (model_datatype) {
  case TensorRT_Byte:
    blob_data_type = DataType::kINT8;
    break;
  case TensorRT_Char:
    blob_data_type = DataType::kINT8;
    break;
  case TensorRT_Int:
    blob_data_type = DataType::kINT32;
    break;
  case TensorRT_Half:
    blob_data_type = DataType::kHALF;
    break;
  case TensorRT_Float:
    blob_data_type = DataType::kFLOAT;
    break;
  default:
    throw std::runtime_error("invalid model datatype");
  }
  return blob_data_type;
}

void setTensorScales(const INetworkDefinition& network, float inScales = 2.0f, float outScales = 4.0f)
{
    // Ensure that all layer inputs have a scale.
    for (int l = 0; l < network.getNbLayers(); l++)
    {
        auto layer = network.getLayer(l);
        for (int i = 0; i < layer->getNbInputs(); i++)
        {
            ITensor* input{layer->getInput(i)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input && !input->dynamicRangeIsSet())
            {
                input->setDynamicRange(-inScales, inScales);
            }
        }
        for (int o = 0; o < layer->getNbOutputs(); o++)
        {
            ITensor* output{layer->getOutput(o)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == LayerType::kPOOLING)
                {
                    output->setDynamicRange(-inScales, inScales);
                }
                else
                {
                    output->setDynamicRange(-outScales, outScales);
                }
            }
        }
    }
}

PredictorHandle NewTensorRTCaffePredictor(char *deploy_file, 
                                          char *weights_file,
                                          TensorRT_DType model_datatype,
                                          char **input_layer_names, 
                                          int32_t num_input_layer_names,
                                          char **output_layer_names, 
                                          int32_t num_output_layer_names,
                                          int32_t batch_size) {

  START_C_DEFINION();
  // Create the builder
  IBuilder *builder = createInferBuilder(gLogger);
  if (builder == nullptr) {
    std::string err =
        std::string("cannot create TensorRT builder for ") + deploy_file;
    throw std::runtime_error(err);
  }
  IBuilderConfig* config = builder->createBuilderConfig();
  const auto explicitBatch = 0U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  
  INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
  DataType blob_data_type = get_blob_data_type(model_datatype);


  auto parser = nvcaffeparser1::createCaffeParser();
  if (parser == nullptr) {
    std::string err =
        std::string("cannot create TensorRT Caffe parser for ") + deploy_file;
    throw std::runtime_error(err);
  }

  const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor =
  parser->parse(deploy_file, weights_file, *network, DataType::kFLOAT);

  std::vector<std::string> input_layer_names_vec{};
  for (int ii = 0; ii < num_input_layer_names; ii++) {
    input_layer_names_vec.emplace_back(input_layer_names[ii]);
  }

  std::vector<std::string> output_layer_names_vec{};
  for (int ii = 0; ii < num_output_layer_names; ii++) {
    output_layer_names_vec.emplace_back(output_layer_names[ii]);
    network->markOutput(*blobNameToTensor->find(output_layer_names[ii]));
  }

  builder->setMaxBatchSize(batch_size);
  config->setMaxWorkspaceSize(36 << 20);

  if (blob_data_type == DataType::kHALF && builder->platformHasFastFp16()) {
    config->setFlag(BuilderFlag::kFP16);
    std::cout << "Currently running in fp16 inference" << std::endl;
  }
  if (blob_data_type == DataType::kINT8 && builder->platformHasFastInt8()) {
    config->setFlag(BuilderFlag::kINT8);
    setTensorScales(*network);
    std::cout << "Currently running in int8 inference" << std::endl;
  }

  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

#ifdef DEBUG
  auto nbBindings = engine->getNbBindings();
  std::cout << __LINE__ << "  >>> " << "Number of bindings: " << nbBindings << std::endl;
  auto inputDims = network->getInput(0)->getDimensions();
  std::cout << __LINE__ << "  >>> " << "InputDims: ";
  for (int i = 0; i < inputDims.nbDims; ++i) {
    std::cout << inputDims.d[i] << " ";
  }
  std::cout << std::endl;
  auto outputDims = network->getOutput(0)->getDimensions();
  std::cout << "outputDims: ";
  for (int i = 0; i < outputDims.nbDims; ++i) {
    std::cout << outputDims.d[i] << " ";
  }
  std::cout << std::endl;
#endif

  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  // IHostMemory *trtModelStream = engine->serialize();
  // std::ofstream p("resnet50_v1.engine", std::ios::binary);
  // p.write((const char*)trtModelStream->data(),trtModelStream->size());
  // p.close();

  IExecutionContext *context = engine->createExecutionContext();
  auto predictor = new Predictor(context, input_layer_names_vec,
                                 output_layer_names_vec, batch_size);
  return (PredictorHandle)predictor;

  END_C_DEFINION(nullptr);
}

std::string readBuffer(std::string const& path)
{
    std::string buffer;
    std::ifstream stream(path.c_str(), std::ios::binary);

    if (stream)
    {
        stream >> std::noskipws;
        copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
    }

    return buffer;
}

PredictorHandle NewTensorRTEnginePredictor(char *engine_file, 
                                           char **input_layer_names, 
                                           int32_t num_input_layer_names,
                                           char **output_layer_names, 
                                           int32_t num_output_layer_names, 
                                           int32_t batch_size) {

  START_C_DEFINION();
  std::cout << "Deserializing..." << std::endl;
  std::string buffer = readBuffer(engine_file);

  std::vector<std::string> input_layer_names_vec{};
  for (int ii = 0; ii < num_input_layer_names; ii++) {
    input_layer_names_vec.emplace_back(input_layer_names[ii]);
  }

  std::vector<std::string> output_layer_names_vec{};
  for (int ii = 0; ii < num_output_layer_names; ii++) {
    output_layer_names_vec.emplace_back(output_layer_names[ii]);
  }

  IRuntime *runtime = createInferRuntime(gLogger);
  ICudaEngine *runtime_engine = runtime->deserializeCudaEngine(
      buffer.data(), buffer.size(), nullptr);

  IExecutionContext *context = runtime_engine->createExecutionContext();
  auto predictor = new Predictor(context, input_layer_names_vec,
                                 output_layer_names_vec, batch_size);
  return (PredictorHandle)predictor;

  END_C_DEFINION(nullptr);
}

PredictorHandle NewTensorRTOnnxPredictor(char *model_file, 
                                         TensorRT_DType model_datatype,
                                         char **input_layer_names, 
                                         int32_t num_input_layer_names,
                                         char **output_layer_names, 
                                         int32_t num_output_layer_names,
                                         int32_t batch_size) {

  START_C_DEFINION();

  // Create the builder
  IBuilder *builder = createInferBuilder(gLogger);
  if (builder == nullptr) {
    std::string err =
        std::string("cannot create tensorrt builder for ") + model_file;
    throw std::runtime_error(err);
  }
  IBuilderConfig* config = builder->createBuilderConfig();
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
  INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
  DataType blob_data_type = get_blob_data_type(model_datatype);

  auto parser = nvonnxparser::createParser(*network, gLogger);
  if (parser == nullptr) {
    std::string err =
        std::string("cannot create tensorrt onnx parser for ") + model_file;
    throw std::runtime_error(err);
  }

  std::vector<std::string> input_layer_names_vec{};
  for (int ii = 0; ii < num_input_layer_names; ii++) {
    input_layer_names_vec.emplace_back(input_layer_names[ii]);
  }

  std::vector<std::string> output_layer_names_vec{};
  for (int ii = 0; ii < num_output_layer_names; ii++) {
    output_layer_names_vec.emplace_back(output_layer_names[ii]);
  }

  parser->parseFromFile(model_file, static_cast<int>(reportSeverity));
  for (int i = 0; i < parser->getNbErrors(); ++i)
	{
		std::cout << parser->getError(i)->desc() << std::endl;
	}

  builder->setMaxBatchSize(batch_size);
  config->setMaxWorkspaceSize(36 << 20);

  if (blob_data_type == DataType::kHALF && builder->platformHasFastFp16()) {
    config->setFlag(BuilderFlag::kFP16);
    std::cout << "Currently running in fp16 inference" << std::endl;
  }
  if (blob_data_type == DataType::kINT8 && builder->platformHasFastInt8()) {
    config->setFlag(BuilderFlag::kINT8);
    setTensorScales(*network);
    std::cout << "Currently running in int8 inference" << std::endl;
  }

  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

#ifdef DEBUG
  auto nbBindings = engine->getNbBindings();
  for (int b = 0; b < nbBindings; ++b) {
    std::cout << "Binding name: " << engine->getBindingName(b) << std::endl;
  }
  std::cout << "number of bindings: " << nbBindings << std::endl;
  auto inputDims = network->getInput(0)->getDimensions();
  std::cout << "inputDims: ";
  for (int i = 0; i < inputDims.nbDims; ++i) {
    std::cout << inputDims.d[i] << " ";
  }
  std::cout << std::endl;
  auto outputDims = network->getOutput(0)->getDimensions();
  std::cout << "outputDims: ";
  for (int i = 0; i < outputDims.nbDims; ++i) {
    std::cout << outputDims.d[i] << " ";
  }
  std::cout << std::endl;
#endif

  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  IExecutionContext *context = engine->createExecutionContext();
  auto predictor = new Predictor(context, input_layer_names_vec,
                                 output_layer_names_vec, batch_size);
  return (PredictorHandle)predictor;

  END_C_DEFINION(nullptr);
}

// nvuffparser::UffInputOrder create_uff_input_order(char *input_order) {
//   nvuffparser::UffInputOrder order;
//   if (input_order == "NCHW") {
//     order = UffInputOrder::kNCHW;
//   } else if (input_order == "NHWC") {
//     order = UffInputOrder::kNHWC;
//   } else if (input_order == "NC") {
//     order = UffInputOrder::kNC;
//   } else {
//     throw std::runtime_error("unsupported input order");
//   }
//   return order;
// }

Dims create_uff_input_dims(int *input_shape) {
  Dims3 dims = Dims3(input_shape[1], input_shape[2], input_shape[3]);
  return dims;
}

PredictorHandle NewTensorRTUffPredictor(char *model_file, 
                                        TensorRT_DType model_datatype,
                                        int **input_shapes,
                                        char **input_layer_names, 
                                        int32_t num_input_layer_names,
                                        char **output_layer_names, 
                                        int32_t num_output_layer_names,
                                        int32_t batch_size) {

  START_C_DEFINION();

  // Create the builder
  IBuilder *builder = createInferBuilder(gLogger);
  if (builder == nullptr) {
    std::string err =
        std::string("cannot create tensorrt builder for ") + model_file;
    throw std::runtime_error(err);
  }
  IBuilderConfig* config = builder->createBuilderConfig();
  const auto explicitBatch = 0U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
  INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
  // builder->setDebugSync(true);
  DataType blob_data_type = get_blob_data_type(model_datatype);

  // Parse the caffe model to populate the network, then set the outputs
  // Create the parser according to the specified model format.
  auto parser = nvuffparser::createUffParser();
  if (parser == nullptr) {
    std::string err =
        std::string("cannot create tensorrt uff parser for ") + model_file;
    throw std::runtime_error(err);
  }

  std::vector<std::string> input_layer_names_vec{};
  for (int ii = 0; ii < num_input_layer_names; ii++) {
#ifdef DEBUG
    std::cout << "Input: " << input_layer_names[ii] << " [" << input_shapes[ii][0] << ", " <<
    input_shapes[ii][1] << ", " << input_shapes[ii][2] << "]" << std::endl;
#endif
    input_layer_names_vec.emplace_back(input_layer_names[ii]);
    // Dims3 input_dim = Dims3(input_shapes[ii][1], input_shapes[ii][2], input_shapes[ii][3]);
    // UffInputOrder input_order = UffInputOrder::kNCHW;
    parser->registerInput(
      input_layer_names[ii], 
      Dims3(input_shapes[ii][1], input_shapes[ii][2], input_shapes[ii][0]), 
      nvuffparser::UffInputOrder::kNHWC);
  }

  std::vector<std::string> output_layer_names_vec{};
  for (int ii = 0; ii < num_output_layer_names; ii++) {
#ifdef DEBUG
    std::cout << "Output: " << output_layer_names[ii] << std::endl;
#endif
    output_layer_names_vec.emplace_back(output_layer_names[ii]);
    parser->registerOutput(output_layer_names[ii]);
  }


  parser->parse(model_file, *network, DataType::kFLOAT);
  builder->setMaxBatchSize(batch_size);
  config->setMaxWorkspaceSize(36 << 20);

  if (blob_data_type == DataType::kHALF && builder->platformHasFastFp16()) {
    config->setFlag(BuilderFlag::kFP16);
    std::cout << "Currently running in fp16 inference" << std::endl;
  }
  if (blob_data_type == DataType::kINT8 && builder->platformHasFastInt8()) {
    config->setFlag(BuilderFlag::kINT8);
    setTensorScales(*network);
    std::cout << "Currently running in int8 inference" << std::endl;
  }

  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

  parser->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();

  IExecutionContext *context = engine->createExecutionContext();
  if (!context) {
  std::cout << "context empty" << std::endl;
  }

  auto predictor = new Predictor(context, input_layer_names_vec,
                                 output_layer_names_vec, batch_size);

  return (PredictorHandle)predictor;

  END_C_DEFINION(nullptr);
}

void TensorRTPredictor_AddInput(PredictorHandle predictor_handle,
                               const char *name, TensorRT_DType dtype,
                               void *host_data, size_t num_elements) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  switch (dtype) {
#define DISPATCH_ADD_INPUT(DType, CType)                                       \
  case DType:                                                                  \
    predictor->AddInput<CType>(name, reinterpret_cast<CType *>(host_data),     \
                               num_elements);                                  \
    break;
    TensorRT_DType_Dispatch(DISPATCH_ADD_INPUT);
#undef DISPATCH_ADD_INPUT
  default:
    throw std::runtime_error("unexpected input type");
  }
  END_C_DEFINION();
}

void TensorRTPredictor_AddOutput(PredictorHandle predictor_handle,
                                const char *name, TensorRT_DType dtype) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  switch (dtype) {
#define DISPATCH_ADD_OUTPUT(DType, CType)                                      \
  case DType:                                                                  \
    predictor->AddOutput<CType>(name);                                         \
    break;
    TensorRT_DType_Dispatch(DISPATCH_ADD_OUTPUT);
#undef DISPATCH_ADD_OUTPUT
  default:
    throw std::runtime_error("unexpected input type");
  }
  END_C_DEFINION();
}

void TensorRTPredictor_Synchronize(PredictorHandle predictor_handle) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  CHECK(predictor->synchronize());
  END_C_DEFINION();
}

void TensorRTPredictor_Run(PredictorHandle predictor_handle) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  predictor->Run();
  END_C_DEFINION();
}

int TensorRTPredictor_GetNumOutputs(PredictorHandle predictor_handle) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  return predictor->output_layer_names_.size();
  END_C_DEFINION(-1);
}

void *TensorRTPredictor_GetOutput(PredictorHandle predictor_handle,
                                 const char *name, int32_t *ndims,
                                 int32_t **res_dims) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);

  auto dims = predictor->GetOutputShape(name);
  void *data = predictor->GetOutputData(name);
  *ndims = dims.size();
#ifdef DEBUG
  std::cout << __LINE__ << "  >>> " << "*ndims = " << *ndims << "\n";
#endif
  *res_dims = (int32_t *)malloc(sizeof(int32_t) * (*ndims));
  memcpy(*res_dims, dims.data(), sizeof(int32_t) * (*ndims));
  return data;
  END_C_DEFINION(nullptr);
}

bool TensorRTPredictor_HasError(PredictorHandle predictor_handle) {
  return has_error;
}

const char *TensorRTPredictor_GetLastError(PredictorHandle predictor_handle) {
  return error_string.c_str();
}

void TensorRTPredictor_Delete(PredictorHandle predictor_handle) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  if (predictor != nullptr) {
    delete predictor;
  }
  END_C_DEFINION();
}

void TensorRTPredictor_StartProfiling(PredictorHandle predictor_handle,
                                     const char *name, const char *metadata) {

  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(predictor_handle);
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  if (predictor->prof_ == nullptr) {
    predictor->prof_ = new profile(name, metadata);
  } else {
    predictor->prof_->reset();
  }
  END_C_DEFINION();
}

void TensorRTPredictor_EndProfiling(PredictorHandle pred) {
  START_C_DEFINION();
  auto predictor = get_predictor_from_handle(pred);
  if (predictor->prof_) {
    predictor->prof_->end();
  }
  END_C_DEFINION();
}

char *TensorRTPredictor_ReadProfiling(PredictorHandle pred) {
  START_C_DEFINION();
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return strdup("");
  }
  if (predictor->prof_ == nullptr) {
    return strdup("");
  }
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
  END_C_DEFINION(nullptr);
}

void TensorRT_Init() { initLibNvInferPlugins(&gLogger, ""); }

#endif // __linux__
