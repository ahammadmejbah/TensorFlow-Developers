#!/bin/bash

# Root directory
mkdir -p TensorFlow_90_Day_Guide

# Phase1_Foundations_of_Python_and_Mathematics
mkdir -p TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics
for day in {1..10}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics/${DIR}/Scripts
    touch TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics/${DIR}/Lecture_Notes.ipynb
    touch TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics/${DIR}/Exercises.ipynb
    touch TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase1_Foundations_of_Python_and_Mathematics/${DIR}/Resources.md
done

# Phase2_Introduction_to_TensorFlow_and_Keras
mkdir -p TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras
for day in {11..20}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras/${DIR}/Scripts
    touch TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras/${DIR}/Lecture_Notes.ipynb
    touch TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras/${DIR}/Exercises.ipynb
    touch TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase2_Introduction_to_TensorFlow_and_Keras/${DIR}/Resources.md
done

# Phase3_Core_Concepts_of_Neural_Networks
mkdir -p TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks
for day in {21..30}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks/${DIR}/Scripts
    touch TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks/${DIR}/Lecture_Notes.ipynb
    touch TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks/${DIR}/Exercises.ipynb
    touch TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase3_Core_Concepts_of_Neural_Networks/${DIR}/Resources.md
done

# Phase4_Advanced_Neural_Network_Architectures
mkdir -p TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures
for day in {31..40}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures/${DIR}/Scripts
    touch TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures/${DIR}/Lecture_Notes.ipynb
    touch TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures/${DIR}/Exercises.ipynb
    touch TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase4_Advanced_Neural_Network_Architectures/${DIR}/Resources.md
done

# Phase5_Specialized_Models_and_Techniques
mkdir -p TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques
for day in {41..50}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques/${DIR}/Scripts
    touch TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques/${DIR}/Lecture_Notes.ipynb
    touch TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques/${DIR}/Exercises.ipynb
    touch TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase5_Specialized_Models_and_Techniques/${DIR}/Resources.md
done

# Phase6_Model_Optimization_and_Deployment
mkdir -p TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment
for day in {51..60}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment/${DIR}/Scripts
    touch TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment/${DIR}/Lecture_Notes.ipynb
    touch TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment/${DIR}/Exercises.ipynb
    touch TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase6_Model_Optimization_and_Deployment/${DIR}/Resources.md
done

# Phase7_Advanced_TensorFlow_Techniques_and_Applications
mkdir -p TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications
for day in {61..70}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications/${DIR}/Scripts
    touch TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications/${DIR}/Lecture_Notes.ipynb
    touch TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications/${DIR}/Exercises.ipynb
    touch TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase7_Advanced_TensorFlow_Techniques_and_Applications/${DIR}/Resources.md
done

# Phase8_Exploring_Various_Applications
mkdir -p TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications
for day in {71..80}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications/${DIR}/Scripts
    touch TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications/${DIR}/Lecture_Notes.ipynb
    touch TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications/${DIR}/Exercises.ipynb
    touch TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase8_Exploring_Various_Applications/${DIR}/Resources.md
done

# Phase9_Capstone_Projects_and_Review
mkdir -p TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review
for day in {81..90}
do
    mkdir -p TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/Day${day}_*
    DIR=$(ls TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review | grep "Day${day}_")
    mkdir -p TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Scripts
    mkdir -p TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Documentation
    mkdir -p TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Resources
    mkdir -p TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Deployment_Scripts
    touch TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/README.md
    touch TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Project_Description.md
    touch TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Data/*.*
    touch TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Notebooks/*.ipynb
    touch TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Scripts/*.py
    touch TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Models/*.*
    touch TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Results/*.txt
    touch TensorFlow_90_Day_Guide/Phase9_Capstone_Projects_and_Review/${DIR}/Resources.md
done

# Additional_Tips_and_Resources
mkdir -p TensorFlow_90_Day_Guide/Additional_Tips_and_Resources
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Consistent_Practice.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Community_Engagement.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Stay_Updated.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Maintain_Documentation.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Seek_Feedback.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Utilize_Version_Control.md

# Projects
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Data/custom_dataset
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Notebooks
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Scripts
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Models/saved_model
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Results
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/README.md
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Notebooks/end_to_end_classification.ipynb
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Scripts/data_preprocessing.py
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Scripts/train_model.py
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Scripts/evaluate_model.py
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Resources.md

mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Data
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Notebooks
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Scripts
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Models/saved_transformer_model
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Results
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/README.md
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Data/text_corpus.txt
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Notebooks/text_generation_transformers.ipynb
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Scripts/preprocess_text.py
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Scripts/train_transformer.py
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Scripts/generate_text.py
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Resources.md

mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Data/object_detection_dataset
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Notebooks
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Models/saved_object_detector
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/API
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Results
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/README.md
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Notebooks/real_time_object_detection.ipynb
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts/prepare_data.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts/train_object_detector.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts/deploy_object_detector.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts/real_time_detection.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/API/app.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/API/requirements.txt
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/API/Dockerfile
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Resources.md

# Data
mkdir -p TensorFlow_90_Day_Guide/Data/MNIST
touch TensorFlow_90_Day_Guide/Data/MNIST/train-images-idx3-ubyte.gz
touch TensorFlow_90_Day_Guide/Data/MNIST/train-labels-idx1-ubyte.gz
touch TensorFlow_90_Day_Guide/Data/MNIST/t10k-images-idx3-ubyte.gz
touch TensorFlow_90_Day_Guide/Data/MNIST/t10k-labels-idx1-ubyte.gz

mkdir -p TensorFlow_90_Day_Guide/Data/CIFAR-10
touch TensorFlow_90_Day_Guide/Data/CIFAR-10/data_batch_1
touch TensorFlow_90_Day_Guide/Data/CIFAR-10/data_batch_2
touch TensorFlow_90_Day_Guide/Data/CIFAR-10/data_batch_3
touch TensorFlow_90_Day_Guide/Data/CIFAR-10/data_batch_4
touch TensorFlow_90_Day_Guide/Data/CIFAR-10/data_batch_5
touch TensorFlow_90_Day_Guide/Data/CIFAR-10/test_batch
touch TensorFlow_90_Day_Guide/Data/CIFAR-10/batches.meta

mkdir -p TensorFlow_90_Day_Guide/Data/IMDb/aclImdb/train
mkdir -p TensorFlow_90_Day_Guide/Data/IMDb/aclImdb/test
mkdir -p TensorFlow_90_Day_Guide/Data/IMDb/aclImdb/unsup
touch TensorFlow_90_Day_Guide/Data/IMDb/aclImdb/train/*
touch TensorFlow_90_Day_Guide/Data/IMDb/aclImdb/test/*
touch TensorFlow_90_Day_Guide/Data/IMDb/aclImdb/unsup/*

mkdir -p TensorFlow_90_Day_Guide/Data/custom_dataset/images
mkdir -p TensorFlow_90_Day_Guide/Data/custom_dataset/annotations
touch TensorFlow_90_Day_Guide/Data/custom_dataset/images/*
touch TensorFlow_90_Day_Guide/Data/custom_dataset/annotations/*

# Models
mkdir -p TensorFlow_90_Day_Guide/Models/saved_model
mkdir -p TensorFlow_90_Day_Guide/Models/saved_transformer_model
mkdir -p TensorFlow_90_Day_Guide/Models/saved_object_detector
mkdir -p TensorFlow_90_Day_Guide/Models/custom_models
touch TensorFlow_90_Day_Guide/Models/saved_model/*
touch TensorFlow_90_Day_Guide/Models/saved_transformer_model/*
touch TensorFlow_90_Day_Guide/Models/saved_object_detector/*
touch TensorFlow_90_Day_Guide/Models/custom_models/*

# Notebooks
mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase1
for day in {1..10}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase1/Day${day}.ipynb
done

mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase2
for day in {11..20}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase2/Day${day}.ipynb
done

mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase3
for day in {21..30}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase3/Day${day}.ipynb
done

mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase4
for day in {31..40}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase4/Day${day}.ipynb
done

mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase5
for day in {41..50}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase5/Day${day}.ipynb
done

mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase6
for day in {51..60}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase6/Day${day}.ipynb
done

mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase7
for day in {61..70}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase7/Day${day}.ipynb
done

mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase8
for day in {71..80}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase8/Day${day}.ipynb
done

mkdir -p TensorFlow_90_Day_Guide/Notebooks/Phase9
for day in {81..90}
do
    touch TensorFlow_90_Day_Guide/Notebooks/Phase9/Day${day}.ipynb
done

touch TensorFlow_90_Day_Guide/Notebooks/Capstone_Project.ipynb

# Scripts
mkdir -p TensorFlow_90_Day_Guide/Scripts/common
touch TensorFlow_90_Day_Guide/Scripts/common/data_loading.py
touch TensorFlow_90_Day_Guide/Scripts/common/utils.py
touch TensorFlow_90_Day_Guide/Scripts/common/visualization.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase1
touch TensorFlow_90_Day_Guide/Scripts/Phase1/basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/data_structures.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/functions_modules.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/oop.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/numpy_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/pandas_manipulation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/data_visualization.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/linear_algebra.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/calculus_ml.py
touch TensorFlow_90_Day_Guide/Scripts/Phase1/probability_statistics.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase2
touch TensorFlow_90_Day_Guide/Scripts/Phase2/install_dependencies.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/tensorflow_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/keras_api_overview.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/first_neural_network.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/compile_train_models.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/evaluate_improve_models.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/data_handling_tf.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/working_with_datasets.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/saving_loading_models.py
touch TensorFlow_90_Day_Guide/Scripts/Phase2/mini_project_train.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase3
touch TensorFlow_90_Day_Guide/Scripts/Phase3/perceptron.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/activation_functions.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/loss_functions.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/optimization_algorithms.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/regularization_techniques.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/normalization.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/dropout.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/feedforward_nn.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/evaluation_metrics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase3/apply_evaluation_metrics.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase4
touch TensorFlow_90_Day_Guide/Scripts/Phase4/cnn_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/cnn_layers.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/cnn_architectures.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/rnn_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/lstm_networks.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/gru_units.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/transformers_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/attention_mechanisms.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/transformer_models.py
touch TensorFlow_90_Day_Guide/Scripts/Phase4/project_cifar10_train.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase5
touch TensorFlow_90_Day_Guide/Scripts/Phase5/autoencoders.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/vae.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/gans.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/conditional_gan.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/style_gan.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/cyclegan.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/nlp_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/sequence_models.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/transformer_nlp.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/sentiment_analysis.py
touch TensorFlow_90_Day_Guide/Scripts/Phase5/project_sentiment_lstm_train.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase6
touch TensorFlow_90_Day_Guide/Scripts/Phase6/hyperparameter_tuning_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/grid_random_search.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/bayesian_optimization.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/advanced_tuning.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/advanced_metrics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/tf_serving_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/deploy_rest_api.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/tensorflow_lite_intro.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/model_compression_quantization.py
touch TensorFlow_90_Day_Guide/Scripts/Phase6/project_deploy_rest_api.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase7
touch TensorFlow_90_Day_Guide/Scripts/Phase7/custom_layers.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/custom_models.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/tfx_overview.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/data_validation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/data_transformation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/model_training.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/model_serving.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/distributed_training.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/multi_gpu_training.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/tpu_training.py
touch TensorFlow_90_Day_Guide/Scripts/Phase7/performance_optimization.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase8
touch TensorFlow_90_Day_Guide/Scripts/Phase8/object_detection_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/yolo_implementation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/ssd_implementation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/faster_rcnn_implementation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/image_segmentation_basics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/unet_segmentation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/mask_rcnn.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/transfer_learning_vision.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/implement_transfer_learning.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/time_series_forecasting.py
touch TensorFlow_90_Day_Guide/Scripts/Phase8/anomaly_detection.py

mkdir -p TensorFlow_90_Day_Guide/Scripts/Phase9
touch TensorFlow_90_Day_Guide/Scripts/Phase9/capstone_project_planning.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/collect_data.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/preprocess_data.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/model_development.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/model_evaluation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/deploy_model.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/prepare_presentation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/finalize_presentation.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/reflection.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/explore_advanced_topics.py
touch TensorFlow_90_Day_Guide/Scripts/Phase9/plan_next_steps.py

# Additional_Tips_and_Resources
mkdir -p TensorFlow_90_Day_Guide/Additional_Tips_and_Resources
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Consistent_Practice.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Community_Engagement.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Stay_Updated.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Maintain_Documentation.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Seek_Feedback.md
touch TensorFlow_90_Day_Guide/Additional_Tips_and_Resources/Utilize_Version_Control.md

# Projects
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Data/custom_dataset
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Notebooks
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Scripts
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Models/saved_model
mkdir -p TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Results
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/README.md
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Notebooks/end_to_end_classification.ipynb
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Scripts/data_preprocessing.py
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Scripts/train_model.py
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Scripts/evaluate_model.py
touch TensorFlow_90_Day_Guide/Projects/Project1_End-to-End_Image_Classification/Resources.md

mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Data
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Notebooks
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Scripts
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Models/saved_transformer_model
mkdir -p TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Results
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/README.md
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Data/text_corpus.txt
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Notebooks/text_generation_transformers.ipynb
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Scripts/preprocess_text.py
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Scripts/train_transformer.py
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Scripts/generate_text.py
touch TensorFlow_90_Day_Guide/Projects/Project2_Text_Generation_with_Transformers/Resources.md

mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Data/object_detection_dataset
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Notebooks
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Models/saved_object_detector
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/API
mkdir -p TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Results
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/README.md
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Notebooks/real_time_object_detection.ipynb
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts/prepare_data.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts/train_object_detector.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts/deploy_object_detector.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Scripts/real_time_detection.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/API/app.py
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/API/requirements.txt
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/API/Dockerfile
touch TensorFlow_90_Day_Guide/Projects/Project3_Real-Time_Object_Detection_System/Resources.md

# Global Files
touch TensorFlow_90_Day_Guide/README.md
touch TensorFlow_90_Day_Guide/LICENSE
touch TensorFlow_90_Day_Guide/Requirements.txt
