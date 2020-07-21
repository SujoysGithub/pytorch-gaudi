#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>

#include <hcl_communicator.h>

namespace c10d {

// Now continue on other work in the current stream.
class ProcessGroupHCL : public ProcessGroup {
 public:
  class WorkHCL : public ProcessGroup::Work {
   public:
    // Constructor takes a list of CUDA devices
    WorkHCL(const std::vector<at::Device>& devices);
    virtual ~WorkHCL();

    bool isCompleted() override;

    bool isSuccess() const override;

    bool wait() override;

    void abort() override;

    void synchronize() override;


   protected:
    std::vector<at::Device> devices_;
    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;
   private:
    friend class ProcessGroupHCL;
  };

  ProcessGroupHCL(
      const std::shared_ptr<Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& opTimeout);

  virtual ~ProcessGroupHCL();
  void abort();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag);

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag);

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag);

  std::shared_ptr<ProcessGroup::Work> barrier(

 private:

  // Helper that encapsulates work shared across all collective communication
  template <typename Fn>
  std::shared_ptr<ProcessGroup::Work> hclcollective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn);
  template <typename Fn, typename PreProcess, typename PostProcess>
  std::shared_ptr<ProcessGroup::Work> hclcollective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      PreProcess pre,
      PostProcess post);

 protected:

  virtual std::shared_ptr<ProcessGroupHCL::WorkHCL> initWork(
      std::vector<at::Device> devices);

  synapse_helpers::hcl_communicator& getComm(int deviceId);
  // Helper function that is called by the destructor
  void destroy();
  bool stop_;

  //Maintains the list of communicators associated with the devices.
  std::map<int, std::unique_ptr<synapse_helpers::hcl_communicator>> hcl_communicator_;
};

} // namespace c10d
