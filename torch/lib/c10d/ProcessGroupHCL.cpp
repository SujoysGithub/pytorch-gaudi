#include <c10d/ProcessGroupHCL.hpp>
#include <map>

using namespace synapse_helpers;

namespace c10d {

namespace {

std::map<at::ScalarType, synDataType> hclDataType = {
   {at::kByte, syn_type_int8},
   {at::kChar, syn_type_int8},
   {at::kDouble, syn_type_na},
   {at::kFloat, syn_type_float},
   {at::kHalf, syn_type_bf16},
   {at::kInt, syn_type_int32},
   {at::kLong, syn_type_na},
   {at::kShort, syn_type_int16},
};

// HCL op mapping
std::map<ReduceOp, HCL_Op> hclOp = {
    {ReduceOp::MIN, eHCLOpNone},
    {ReduceOp::MAX, eHCLOpNone},
    {ReduceOp::SUM, eHCLSum},
    {ReduceOp::PRODUCT, eHCLMul},
};

HCL_Op getHCLOpType(ReduceOp type) {
  try {
    return hclOp.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported operation type for HCL process group");
  }
}

synDataType getHCLDataType(at::ScalarType type) {
  try {
    return hclDataType.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for HCL process group");
  }
}

}

hcl_communicator& ProcessGroupHCL::getComm(int deviceId) {
  if (hcl_communicator_.find(deviceId) == hcl_communicator_.end())
  {
    char* config_json_path = std::getenv("HCL_CONFIG_PATH");
    if (!config_json_path) {
      LOG(FATAL) << "Please export HCL_CONFIG_PATH...";
    }
    HCL_Comm pgComm_ = HCL_COMM_WORLD;
    hcl_communicator_[deviceId] = std::make_unique<hcl_communicator>(deviceId, pgComm_, config_json_path);
  }
  return *(hcl_communicator_.find(deviceId)->second);
}
// TBD: Store not used for now and config done from file
// Initial support added for multiple devices on a single node
// So using rank as the device id.  This will be enhanced further.
ProcessGroupHCL::ProcessGroupHCL(
    const std::shared_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& opTimeout)
    : ProcessGroup(rank, size), stop_(false) {

}

ProcessGroupHCL::~ProcessGroupHCL() {
  destroy();
}

void ProcessGroupHCL::destroy() {

}

void ProcessGroupHCL::abort() {
  destroy();

}


ProcessGroupHCL::WorkHCL::WorkHCL(const std::vector<at::Device>& devices)
    : devices_(devices), workStartTime_(std::chrono::steady_clock::now()) {

 // cudaEvents_.resize(devices.size());
 // ncclComms_.resize(devices.size());
}
ProcessGroupHCL::WorkHCL::~WorkHCL() {}


ProcessGroupHCL::WorkHCL::WorkHCL(const std::vector<at::Device>& devices)
    : devices_(devices), workStartTime_(std::chrono::steady_clock::now()) {

 // cudaEvents_.resize(devices.size());
 // ncclComms_.resize(devices.size());
}
ProcessGroupHCL::WorkHCL::~WorkHCL() {}


bool ProcessGroupHCL::WorkHCL::isCompleted() {

  return exception() || true; //check for the completion of work;
}

bool ProcessGroupHCL::WorkHCL::isSuccess() const {
  if (exception()) {
    // Already detected an exception.
    return false;
  }

  return true;
}

// Same as calling synchronize().
bool ProcessGroupHCL::WorkHCL::wait() {
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupHCL::WorkHCL::synchronize() {

}

void ProcessGroupHCL::WorkHCL::abort() {
  TORCH_CHECK(false, "ProcessGroupHCL::WorkHCL::abort not implemented.");
}


  td::vector<at::Device> devices) {
  return std::make_shared<ProcessGroupHCL::WorkHCL>(devices);
}


// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}


template <typename Fn, typename PreProcess, typename PostProcess>
std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::hclcollective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post){

  const auto devices = getDeviceList(inputs);
  auto work = initWork(devices);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& hcl_comm = getComm(inputs[i].get_device());
    fn(inputs[i], outputs[i], hcl_comm);
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
        // Update work
  }
  return work;
}


template <typename Fn>
std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::hclcollective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn) {

    //Need to replace int by device work streams
  return hclcollective(
      inputs,
      outputs,
      fn,
      [](std::vector<int>&) {},
      [](std::vector<int>&) {});

}



std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {

  //std::cout << "Broadcast called for rank " << getRank() << std::endl;
  return hclcollective(
    tensors,
    tensors,
    [&](at::Tensor& input,
    at::Tensor& output,
    hcl_communicator& hcl_comm) {
                                    return hcl_comm.broadcast(opts.rootRank,
                                    (synapse_helpers::device_ptr)input.data_ptr(),
                                    input.numel(),
                                    getHCLDataType(input.scalar_type()));
                                });

}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {

  //std::cout << "Allreduce called for rank " << getRank() << std::endl;
  return hclcollective(
 ensors,
    [&](at::Tensor& input,
    at::Tensor& output,
    hcl_communicator& hcl_comm) {
                                    return hcl_comm.allreduce((synapse_helpers::device_ptr)input.data_ptr(),
                                    (synapse_helpers::device_ptr)input.data_ptr(),
                                    input.numel(),
                                    getHCLDataType(input.scalar_type()));
                                });
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with HCL");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {

  return hclcollective(
    tensors,
    tensors,
    [&](at::Tensor& input,
    at::Tensor& output,
    hcl_communicator& hcl_comm) {
                                    return hcl_comm.reduce(opts.rootRank,
                                    (synapse_helpers::device_ptr)input.data_ptr(),
                                    (synapse_helpers::device_ptr)output.data_ptr(),
                                    input.numel(),
                                    getHCLDataType(input.scalar_type()),
                                    getHCLOpType(opts.reduceOp));
                                });
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  throw std::runtime_error(
      "allgather is currently not supported with HCL");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts) {
  throw std::runtime_error(
      "allgather_base is currently not supported with HCL");
}


std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupHCL does not support allgather_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupHCL does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  throw std::runtime_error(
      "ProcessGroupHCL does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupHCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {

  return hclcollective(
    tensors,
    tensors,
    [&](at::Tensor& input,
    at::Tensor& output,
    hcl_communicator& hcl_comm) {
                                    return hcl_comm.send((synapse_helpers::device_ptr)input.data_ptr(),
                                    input.numel() * sizeof(getHCLDataType(input.scalar_type())),
                                    dstRank,
                                    tag);
                                });
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {

  return hclcollective(
    tensors,
    tensors,
    [&](at::Tensor& input,
    at::Tensor& output,
    hcl_communicator& hcl_comm) {
                                    return hcl_comm.receive((synapse_helpers::device_ptr)input.data_ptr(),
                                    input.numel() * sizeof(getHCLDataType(input.scalar_type())),
                                    srcRank,
                                    tag);
                                });
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error(
      "ProcessGroupHCL does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupHCL::barrier(
    const BarrierOptions& opts) {

  throw std::runtime_error(
      "ProcessGroupHCL does not support barrier");
}

} // namespace c10d
