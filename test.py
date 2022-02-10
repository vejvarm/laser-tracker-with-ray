import tensorflow as tf


def gpu_test():
    return True if len(tf.config.list_physical_devices('GPU')) > 0 else False


if __name__ == '__main__':
    print(gpu_test())
    is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    print(is_cuda_gpu_available)