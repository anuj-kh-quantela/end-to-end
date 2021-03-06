cmake_minimum_required(VERSION 3.7)
project(darknet)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/activation_layer.c
        src/activation_layer.h
        src/activations.c
        src/activations.h
        src/art.c
        src/avgpool_layer.c
        src/avgpool_layer.h
        src/batchnorm_layer.c
        src/batchnorm_layer.h
        src/blas.c
        src/blas.h
        src/box.c
        src/box.h
        src/captcha.c
        src/cifar.c
        src/classifier.c
        src/classifier.h
        src/coco.c
        src/col2im.c
        src/col2im.h
        src/compare.c
        src/connected_layer.c
        src/connected_layer.h
        src/convolutional_layer.c
        src/convolutional_layer.h
        src/cost_layer.c
        src/cost_layer.h
        src/crnn_layer.c
        src/crnn_layer.h
        src/crop_layer.c
        src/crop_layer.h
        src/cuda.c
        src/cuda.h
        src/darknet.c
        src/data.c
        src/data.h
        src/deconvolutional_layer.c
        src/deconvolutional_layer.h
        src/demo.c
        src/demo.h
        src/detection_layer.c
        src/detection_layer.h
        src/detector.c
        src/dice.c
        src/dropout_layer.c
        src/dropout_layer.h
        src/gemm.c
        src/gemm.h
        src/go.c
        src/gru_layer.c
        src/gru_layer.h
        src/im2col.c
        src/im2col.h
        src/image.c
        src/image.h
        src/layer.c
        src/layer.h
        src/list.c
        src/list.h
        src/local_layer.c
        src/local_layer.h
        src/lsd.c
        src/matrix.c
        src/matrix.h
        src/maxpool_layer.c
        src/maxpool_layer.h
        src/network.c
        src/network.h
        src/nightmare.c
        src/normalization_layer.c
        src/normalization_layer.h
        src/option_list.c
        src/option_list.h
        src/parser.c
        src/parser.h
        src/region_layer.c
        src/region_layer.h
        src/regressor.c
        src/reorg_layer.c
        src/reorg_layer.h
        src/rnn.c
        src/rnn_layer.c
        src/rnn_layer.h
        src/rnn_vid.c
        src/route_layer.c
        src/route_layer.h
        src/shortcut_layer.c
        src/shortcut_layer.h
        src/softmax_layer.c
        src/softmax_layer.h
        src/stb_image.h
        src/stb_image_write.h
        src/super.c
        src/swag.c
        src/tag.c
        src/tree.c
        src/tree.h
        src/utils.c
        src/utils.h
        src/voxel.c
        src/writing.c
        src/yolo.c)

add_executable(darknet ${SOURCE_FILES})