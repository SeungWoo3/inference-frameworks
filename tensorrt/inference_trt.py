import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

    def load_engine(self, engine_path):
        """TensorRT 엔진 로드"""
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        """입출력 버퍼 할당"""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for index in range(self.engine.num_io_tensors):  # TensorRT 8.5+에서 사용해야 함
            tensor_name = self.engine.get_tensor_name(index)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size = np.prod(shape)

            # GPU 메모리 할당
            device_mem = cuda.mem_alloc(int(size * np.dtype(dtype).itemsize))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(device_mem)
            else:
                self.outputs.append(device_mem)

            self.bindings.append(int(device_mem))

    def infer(self, input_data):
        """FP16 엔진에서 추론 실행"""
        if input_data.dtype != np.float16:
            input_data = input_data.astype(np.float16)

        input_name = self.engine.get_tensor_name(0)  # 입력 텐서 이름 가져오기
        expected_shape = self.engine.get_tensor_shape(input_name)  # 입력 텐서의 shape 가져오기

        # 입력 크기 확인 및 변환
        if input_data.shape != tuple(expected_shape):
            input_data = np.resize(input_data, tuple(expected_shape))

        # GPU 메모리 복사
        cuda.memcpy_htod(self.inputs[0], input_data)

        # 실행
        self.context.execute_v2(self.bindings)

        # 출력 텐서 가져오기
        output_name = self.engine.get_tensor_name(1)
        output_shape = self.engine.get_tensor_shape(output_name)

        # 출력 메모리 할당
        output_data = np.empty(tuple(output_shape), dtype=np.float16)
        
        if len(self.outputs) > 0:
            cuda.memcpy_dtoh(output_data, self.outputs[0])
        else:
            raise RuntimeError("Output buffer is empty. Check TensorRT bindings.")

        return output_data


# ==================== 실행 부분 ====================

# 모델 로드
simple_trt = TRTInference("simple_model.trt")
vit_trt = TRTInference("vit_model.trt")

# 입력 데이터 (각 모델에 맞게 수정 필요)
input_shape_simple = (1, 3, 224, 224)  # 예시 입력 크기
input_shape_vit = (1, 3, 384, 384)  # 예시 입력 크기

input_data_simple = np.random.randn(*input_shape_simple).astype(np.float32)
input_data_vit = np.random.randn(*input_shape_vit).astype(np.float32)

# 추론 실행
start_time = time.time()
output_simple = simple_trt.infer(input_data_simple)
output_vit = vit_trt.infer(input_data_vit)
end_time = time.time()

# 결과 출력
print(f"Simple Model Output: {output_simple[:5]}")  # 일부 출력
print(f"ViT Model Output: {output_vit[:5]}")  # 일부 출력
print(f"Inference Time: {end_time - start_time:.4f} sec")
