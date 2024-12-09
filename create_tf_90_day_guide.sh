#!/bin/bash

# Root directory
ROOT_DIR="TensorFlow_90_Day_Guide"
mkdir -p "$ROOT_DIR"

# Phase1_Foundations_of_Python_and_Mathematics
mkdir -p "$ROOT_DIR/Phase1_Foundations_of_Python_and_Mathematics"

declare -a phase1_days=(
    "Day1_Python_Basics_Refresher"
    "Day2_Python_Data_Structures"
    "Day3_Functions_and_Modules"
    "Day4_Object_Oriented_Programming"
    "Day5_Introduction_to_NumPy"
    "Day6_Data_Manipulation_with_Pandas"
    "Day7_Data_Visualization_with_Matplotlib_and_Seaborn"
    "Day8_Linear_Algebra_Fundamentals"
    "Day9_Calculus_for_Machine_Learning"
    "Day10_Probability_and_Statistics_Basics"
)

for day in "${phase1_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase1_Foundations_of_Python_and_Mathematics/$day/Scripts"
    touch "$ROOT_DIR/Phase1_Foundations_of_Python_and_Mathematics/$day/README.md"
    touch "$ROOT_DIR/Phase1_Foundations_of_Python_and_Mathematics/$day/Lecture_Notes.ipynb"
    touch "$ROOT_DIR/Phase1_Foundations_of_Python_and_Mathematics/$day/Exercises.ipynb"
    touch "$ROOT_DIR/Phase1_Foundations_of_Python_and_Mathematics/$day/Scripts/${day}.py"
    touch "$ROOT_DIR/Phase1_Foundations_of_Python_and_Mathematics/$day/Resources.md"
done

# Phase2_Introduction_to_TensorFlow_and_Keras
mkdir -p "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras"

declare -a phase2_days=(
    "Day11_Setting_Up_the_Environment"
    "Day12_TensorFlow_Basics"
    "Day13_Keras_API_Overview"
    "Day14_Building_Your_First_Neural_Network"
    "Day15_Compiling_and_Training_Models"
    "Day16_Evaluating_and_Improving_Models"
    "Day17_Data_Handling_with_TensorFlow"
    "Day18_Working_with_Datasets"
    "Day19_Saving_and_Loading_Models"
    "Day20_Mini_Project_Handwritten_Digit_Classification"
)

for day in "${phase2_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras/$day/Scripts"
    touch "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras/$day/README.md"
    touch "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras/$day/Lecture_Notes.ipynb"
    touch "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras/$day/Exercises.ipynb"
    case "$day" in
        "Day20_Mini_Project_Handwritten_Digit_Classification")
            touch "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras/$day/Scripts/train_model.py"
            touch "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras/$day/Scripts/evaluate_model.py"
            ;;
        *)
            script_name="${day}.py"
            touch "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras/$day/Scripts/${script_name}"
            ;;
    esac
    touch "$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras/$day/Resources.md"
done

# Phase3_Core_Concepts_of_Neural_Networks
mkdir -p "$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks"

declare -a phase3_days=(
    "Day21_Understanding_Perceptrons_and_Neurons"
    "Day22_Activation_Functions_in_Depth"
    "Day23_Loss_Functions_Explained"
    "Day24_Optimization_Algorithms"
    "Day25_Regularization_Techniques"
    "Day26_Batch_Normalization_and_Layer_Normalization"
    "Day27_Introduction_to_Dropout"
    "Day28_Review_and_Implement_a_Feedforward_Neural_Network"
    "Day29_Introduction_to_Model_Evaluation_Metrics"
    "Day30_Hands-On_with_Evaluation_Metrics"
)

for day in "${phase3_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks/$day/Scripts"
    touch "$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks/$day/README.md"
    touch "$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks/$day/Lecture_Notes.ipynb"
    touch "$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks/$day/Exercises.ipynb"
    case "$day" in
        "Day28_Review_and_Implement_a_Feedforward_Neural_Network")
            touch "$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks/$day/Scripts/feedforward_nn.py"
            ;;
        *)
            script_name="${day}.py"
            touch "$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks/$day/Scripts/${script_name}"
            ;;
    esac
    touch "$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks/$day/Resources.md"
done

# Phase4_Advanced_Neural_Network_Architectures
mkdir -p "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures"

declare -a phase4_days=(
    "Day31_Introduction_to_Convolutional_Neural_Networks"
    "Day32_Deep_Dive_into_CNN_Layers"
    "Day33_Architectures_of_CNNs"
    "Day34_Introduction_to_Recurrent_Neural_Networks"
    "Day35_Long_Short-Term_Memory_Networks"
    "Day36_Gated_Recurrent_Units"
    "Day37_Introduction_to_Transformers_and_Attention_Mechanisms"
    "Day38_Implementing_Attention_Mechanisms"
    "Day39_Transformer_Models_for_NLP"
    "Day40_Project_Image_Classification_with_CNNs_on_CIFAR-10"
)

for day in "${phase4_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures/$day/Scripts"
    touch "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures/$day/README.md"
    touch "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures/$day/Lecture_Notes.ipynb"
    touch "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures/$day/Exercises.ipynb"
    case "$day" in
        "Day40_Project_Image_Classification_with_CNNs_on_CIFAR-10")
            touch "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures/$day/Scripts/train_cifar10_cnn.py"
            touch "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures/$day/Scripts/evaluate_cifar10_cnn.py"
            ;;
        *)
            script_name="${day}.py"
            touch "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures/$day/Scripts/${script_name}"
            ;;
    esac
    touch "$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures/$day/Resources.md"
done

# Phase5_Specialized_Models_and_Techniques
mkdir -p "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques"

declare -a phase5_days=(
    "Day41_Introduction_to_Autoencoders"
    "Day42_Variational_Autoencoders"
    "Day43_Generative_Adversarial_Networks"
    "Day44_Advanced_GAN_Techniques"
    "Day45_Introduction_to_NLP_with_TensorFlow"
    "Day46_Sequence_Models_for_NLP"
    "Day47_Transformer_Models_for_NLP"
    "Day48_Text_Classification_and_Sentiment_Analysis"
    "Day49_Project_Sentiment_Analysis_Using_LSTM"
    "Day50_Review_and_Consolidation"
)

for day in "${phase5_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Scripts"
    touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/README.md"
    touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Lecture_Notes.ipynb"
    touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Exercises.ipynb"
    case "$day" in
        "Day44_Advanced_GAN_Techniques")
            touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Scripts/conditional_gan.py"
            touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Scripts/style_gan.py"
            touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Scripts/cyclegan.py"
            ;;
        "Day49_Project_Sentiment_Analysis_Using_LSTM")
            touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Scripts/train_sentiment_lstm.py"
            touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Scripts/evaluate_sentiment_lstm.py"
            ;;
        *)
            script_name="${day}.py"
            touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Scripts/${script_name}"
            ;;
    esac
    touch "$ROOT_DIR/Phase5_Specialized_Models_and_Techniques/$day/Resources.md"
done

# Phase6_Model_Optimization_and_Deployment
mkdir -p "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment"

declare -a phase6_days=(
    "Day51_Hyperparameter_Tuning_Basics"
    "Day52_Grid_Search_and_Random_Search"
    "Day53_Bayesian_Optimization_for_Hyperparameter_Tuning"
    "Day54_Advanced_Hyperparameter_Tuning_Techniques"
    "Day55_Model_Evaluation_Metrics_Revisited"
    "Day56_TensorFlow_Serving_and_Model_Deployment_Basics"
    "Day57_Deploying_Models_as_REST_APIs"
    "Day58_Introduction_to_TensorFlow_Lite"
    "Day59_Model_Compression_and_Quantization"
    "Day60_Project_Deploy_a_Trained_Model_as_a_REST_API"
)

for day in "${phase6_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/Scripts"
    touch "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/README.md"
    touch "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/Lecture_Notes.ipynb"
    touch "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/Exercises.ipynb"
    case "$day" in
        "Day60_Project_Deploy_a_Trained_Model_as_a_REST_API")
            touch "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/Scripts/train_cifar10_cnn.py"
            touch "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/Scripts/deploy_rest_api.py"
            touch "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/Scripts/evaluate_deployment.py"
            ;;
        *)
            script_name="${day}.py"
            touch "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/Scripts/${script_name}"
            ;;
    esac
    touch "$ROOT_DIR/Phase6_Model_Optimization_and_Deployment/$day/Resources.md"
done

# Phase7_Advanced_TensorFlow_Techniques_and_Applications
mkdir -p "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications"

declare -a phase7_days=(
    "Day61_Custom_Layers_in_Keras"
    "Day62_Custom_Models_in_Keras"
    "Day63_TensorFlow_Extended_TFX_Overview"
    "Day64_Data_Validation_and_Transformation_with_TFX"
    "Day65_Model_Training_and_Serving_with_TFX"
    "Day66_Distributed_Training_with_TensorFlow"
    "Day67_Multi-GPU_Training_Strategies"
    "Day68_TPU_Training_with_TensorFlow"
    "Day69_Optimizing_TensorFlow_Performance"
    "Day70_Review_and_Practice_with_Advanced_TensorFlow_Features"
)

for day in "${phase7_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Scripts"
    touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/README.md"
    touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Lecture_Notes.ipynb"
    touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Exercises.ipynb"
    case "$day" in
        "Day64_Data_Validation_and_Transformation_with_TFX")
            touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Scripts/data_validation.py"
            touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Scripts/data_transformation.py"
            ;;
        "Day65_Model_Training_and_Serving_with_TFX")
            touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Scripts/model_training.py"
            touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Scripts/model_serving.py"
            ;;
        *)
            script_names=($(echo "$day" | awk -F'_' '{for(i=1;i<=NF;i++) printf "%s_", $i}'))
            script_names=("${script_names[@]%_}")
            for script in "${script_names[@]}"; do
                touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Scripts/${script}.py"
            done
            ;;
    esac
    touch "$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications/$day/Resources.md"
done

# Phase8_Exploring_Various_Applications
mkdir -p "$ROOT_DIR/Phase8_Exploring_Various_Applications"

declare -a phase8_days=(
    "Day71_Introduction_to_Object_Detection"
    "Day72_Implementing_YOLO_with_TensorFlow"
    "Day73_Implementing_SSD_and_Faster_R-CNN"
    "Day74_Image_Segmentation_Basics"
    "Day75_Implementing_U-Net_for_Image_Segmentation"
    "Day76_Implementing_Mask_R-CNN"
    "Day77_Transfer_Learning_for_Vision_Tasks"
    "Day78_Implementing_Transfer_Learning"
    "Day79_Introduction_to_Time_Series_Forecasting"
    "Day80_Anomaly_Detection_Techniques"
)

for day in "${phase8_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Scripts"
    touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/README.md"
    touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Lecture_Notes.ipynb"
    touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Exercises.ipynb"
    case "$day" in
        "Day73_Implementing_SSD_and_Faster_R-CNN")
            touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Scripts/ssd_implementation.py"
            touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Scripts/faster_rcnn_implementation.py"
            ;;
        "Day78_Implementing_Transfer_Learning")
            touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Scripts/implement_transfer_learning.py"
            ;;
        "Day80_Anomaly_Detection_Techniques")
            touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Scripts/anomaly_detection_autoencoder.py"
            touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Scripts/anomaly_detection_isolation_forest.py"
            ;;
        *)
            script_names=($(echo "$day" | awk -F'_' '{for(i=1;i<=NF;i++) printf "%s_", $i}'))
            script_names=("${script_names[@]%_}")
            for script in "${script_names[@]}"; do
                touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Scripts/${script}.py"
            done
            ;;
    esac
    touch "$ROOT_DIR/Phase8_Exploring_Various_Applications/$day/Resources.md"
done

# Phase9_Capstone_Projects_and_Review
mkdir -p "$ROOT_DIR/Phase9_Capstone_Projects_and_Review"

declare -a phase9_days=(
    "Day81_Capstone_Project_Planning"
    "Day82_Data_Collection_and_Preprocessing"
    "Day83_Model_Development_and_Training"
    "Day84_Model_Evaluation_and_Optimization"
    "Day85_Model_Deployment_and_Documentation"
    "Day86_Presentation_Preparation"
    "Day87_Finalize_and_Present_the_Capstone_Project"
    "Day88_Review_and_Reflect_on_Learning_Journey"
    "Day89_Explore_Advanced_Topics_or_Certifications"
    "Day90_Plan_Next_Steps_and_Continued_Learning"
)

for day in "${phase9_days[@]}"; do
    mkdir -p "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Scripts"
    mkdir -p "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Documentation"
    mkdir -p "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Resources"
    mkdir -p "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Deployment_Scripts"
    touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/README.md"
    case "$day" in
        "Day81_Capstone_Project_Planning")
            touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Project_Selection_Guide.md"
            touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Project_Ideas.md"
            ;;
        "Day85_Model_Deployment_and_Documentation")
            touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Deployment_Scripts/deploy_model.py"
            touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Documentation/Project_Documentation.md"
            touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Documentation/Deployment_Guide.md"
            ;;
        "Day87_Finalize_and_Present_the_Capstone_Project")
            mkdir -p "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Final_Presentation"
            mkdir -p "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Feedback"
            touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Final_Presentation/Final_Presentation.pptx"
            touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Final_Presentation/Presentation_Video.mp4"
            touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Feedback/feedback_notes.md"
            ;;
        *)
            script_names=($(echo "$day" | awk -F'_' '{for(i=1;i<=NF;i++) printf "%s_", $i}'))
            script_names=("${script_names[@]%_}")
            for script in "${script_names[@]}"; do
                touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Scripts/${script}.py"
            done
            ;;
    esac
    touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Project_Description.md"
    touch "$ROOT_DIR/Phase9_Capstone_Projects_and_Review/$day/Resources.md"
done

# Additional_Tips_and_Resources
mkdir -p "$ROOT_DIR/Additional_Tips_and_Resources"
declare -a tips_files=(
    "Consistent_Practice.md"
    "Community_Engagement.md"
    "Stay_Updated.md"
    "Maintain_Documentation.md"
    "Seek_Feedback.md"
    "Utilize_Version_Control.md"
)
for file in "${tips_files[@]}"; do
    touch "$ROOT_DIR/Additional_Tips_and_Resources/$file"
done

# Projects
mkdir -p "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Data/custom_dataset"
mkdir -p "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Notebooks"
mkdir -p "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Scripts"
mkdir -p "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Models/saved_model"
mkdir -p "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Results"
touch "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/README.md"
touch "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Notebooks/end_to_end_classification.ipynb"
touch "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Scripts/data_preprocessing.py"
touch "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Scripts/train_model.py"
touch "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Scripts/evaluate_model.py"
touch "$ROOT_DIR/Projects/Project1_End-to-End_Image_Classification/Resources.md"

mkdir -p "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Data"
mkdir -p "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Notebooks"
mkdir -p "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Scripts"
mkdir -p "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Models/saved_transformer_model"
mkdir -p "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Results"
touch "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/README.md"
touch "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Data/text_corpus.txt"
touch "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Notebooks/text_generation_transformers.ipynb"
touch "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Scripts/preprocess_text.py"
touch "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Scripts/train_transformer.py"
touch "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Scripts/generate_text.py"
touch "$ROOT_DIR/Projects/Project2_Text_Generation_with_Transformers/Resources.md"

mkdir -p "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Data/object_detection_dataset"
mkdir -p "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Notebooks"
mkdir -p "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Scripts"
mkdir -p "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Models/saved_object_detector"
mkdir -p "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/API"
mkdir -p "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Results"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/README.md"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Notebooks/real_time_object_detection.ipynb"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Scripts/prepare_data.py"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Scripts/train_object_detector.py"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Scripts/deploy_object_detector.py"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Scripts/real_time_detection.py"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/API/app.py"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/API/requirements.txt"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/API/Dockerfile"
touch "$ROOT_DIR/Projects/Project3_Real-Time_Object_Detection_System/Resources.md"

# Data Directory
mkdir -p "$ROOT_DIR/Data/MNIST"
touch "$ROOT_DIR/Data/MNIST/train-images-idx3-ubyte.gz"
touch "$ROOT_DIR/Data/MNIST/train-labels-idx1-ubyte.gz"
touch "$ROOT_DIR/Data/MNIST/t10k-images-idx3-ubyte.gz"
touch "$ROOT_DIR/Data/MNIST/t10k-labels-idx1-ubyte.gz"

mkdir -p "$ROOT_DIR/Data/CIFAR-10"
touch "$ROOT_DIR/Data/CIFAR-10/data_batch_1"
touch "$ROOT_DIR/Data/CIFAR-10/data_batch_2"
touch "$ROOT_DIR/Data/CIFAR-10/data_batch_3"
touch "$ROOT_DIR/Data/CIFAR-10/data_batch_4"
touch "$ROOT_DIR/Data/CIFAR-10/data_batch_5"
touch "$ROOT_DIR/Data/CIFAR-10/test_batch"
touch "$ROOT_DIR/Data/CIFAR-10/batches.meta"

mkdir -p "$ROOT_DIR/Data/IMDb/aclImdb/train"
mkdir -p "$ROOT_DIR/Data/IMDb/aclImdb/test"
mkdir -p "$ROOT_DIR/Data/IMDb/aclImdb/unsup"
touch "$ROOT_DIR/Data/IMDb/aclImdb/train/"*
touch "$ROOT_DIR/Data/IMDb/aclImdb/test/"*
touch "$ROOT_DIR/Data/IMDb/aclImdb/unsup/"*

mkdir -p "$ROOT_DIR/Data/custom_dataset/images"
mkdir -p "$ROOT_DIR/Data/custom_dataset/annotations"
touch "$ROOT_DIR/Data/custom_dataset/images/"*
touch "$ROOT_DIR/Data/custom_dataset/annotations/"*

# Models Directory
mkdir -p "$ROOT_DIR/Models/saved_model"
mkdir -p "$ROOT_DIR/Models/saved_transformer_model"
mkdir -p "$ROOT_DIR/Models/saved_object_detector"
mkdir -p "$ROOT_DIR/Models/custom_models"
touch "$ROOT_DIR/Models/saved_model/"*
touch "$ROOT_DIR/Models/saved_transformer_model/"*
touch "$ROOT_DIR/Models/saved_object_detector/"*
touch "$ROOT_DIR/Models/custom_models/"*

# Notebooks Directory
mkdir -p "$ROOT_DIR/Notebooks/Phase1"
mkdir -p "$ROOT_DIR/Notebooks/Phase2"
mkdir -p "$ROOT_DIR/Notebooks/Phase3"
mkdir -p "$ROOT_DIR/Notebooks/Phase4"
mkdir -p "$ROOT_DIR/Notebooks/Phase5"
mkdir -p "$ROOT_DIR/Notebooks/Phase6"
mkdir -p "$ROOT_DIR/Notebooks/Phase7"
mkdir -p "$ROOT_DIR/Notebooks/Phase8"
mkdir -p "$ROOT_DIR/Notebooks/Phase9"
touch "$ROOT_DIR/Notebooks/Phase1/"*.ipynb
touch "$ROOT_DIR/Notebooks/Phase2/"*.ipynb
touch "$ROOT_DIR/Notebooks/Phase3/"*.ipynb
touch "$ROOT_DIR/Notebooks/Phase4/"*.ipynb
touch "$ROOT_DIR/Notebooks/Phase5/"*.ipynb
touch "$ROOT_DIR/Notebooks/Phase6/"*.ipynb
touch "$ROOT_DIR/Notebooks/Phase7/"*.ipynb
touch "$ROOT_DIR/Notebooks/Phase8/"*.ipynb
touch "$ROOT_DIR/Notebooks/Phase9/"*.ipynb
touch "$ROOT_DIR/Notebooks/Capstone_Project.ipynb"

# Scripts Directory
mkdir -p "$ROOT_DIR/Scripts/common"
touch "$ROOT_DIR/Scripts/common/data_loading.py"
touch "$ROOT_DIR/Scripts/common/utils.py"
touch "$ROOT_DIR/Scripts/common/visualization.py"

mkdir -p "$ROOT_DIR/Scripts/Phase1"
touch "$ROOT_DIR/Scripts/Phase1/basics.py"
touch "$ROOT_DIR/Scripts/Phase1/data_structures.py"
touch "$ROOT_DIR/Scripts/Phase1/functions_modules.py"
touch "$ROOT_DIR/Scripts/Phase1/oop.py"
touch "$ROOT_DIR/Scripts/Phase1/numpy_basics.py"
touch "$ROOT_DIR/Scripts/Phase1/pandas_manipulation.py"
touch "$ROOT_DIR/Scripts/Phase1/data_visualization.py"
touch "$ROOT_DIR/Scripts/Phase1/linear_algebra.py"
touch "$ROOT_DIR/Scripts/Phase1/calculus_ml.py"
touch "$ROOT_DIR/Scripts/Phase1/probability_statistics.py"

mkdir -p "$ROOT_DIR/Scripts/Phase2"
touch "$ROOT_DIR/Scripts/Phase2/install_dependencies.py"
touch "$ROOT_DIR/Scripts/Phase2/tensorflow_basics.py"
touch "$ROOT_DIR/Scripts/Phase2/keras_api_overview.py"
touch "$ROOT_DIR/Scripts/Phase2/first_neural_network.py"
touch "$ROOT_DIR/Scripts/Phase2/compile_train_models.py"
touch "$ROOT_DIR/Scripts/Phase2/evaluate_improve_models.py"
touch "$ROOT_DIR/Scripts/Phase2/data_handling_tf.py"
touch "$ROOT_DIR/Scripts/Phase2/working_with_datasets.py"
touch "$ROOT_DIR/Scripts/Phase2/saving_loading_models.py"
touch "$ROOT_DIR/Scripts/Phase2/mini_project_train.py"

mkdir -p "$ROOT_DIR/Scripts/Phase3"
touch "$ROOT_DIR/Scripts/Phase3/perceptron.py"
touch "$ROOT_DIR/Scripts/Phase3/activation_functions.py"
touch "$ROOT_DIR/Scripts/Phase3/loss_functions.py"
touch "$ROOT_DIR/Scripts/Phase3/optimization_algorithms.py"
touch "$ROOT_DIR/Scripts/Phase3/regularization_techniques.py"
touch "$ROOT_DIR/Scripts/Phase3/normalization.py"
touch "$ROOT_DIR/Scripts/Phase3/dropout.py"
touch "$ROOT_DIR/Scripts/Phase3/feedforward_nn.py"
touch "$ROOT_DIR/Scripts/Phase3/evaluation_metrics.py"
touch "$ROOT_DIR/Scripts/Phase3/apply_evaluation_metrics.py"

mkdir -p "$ROOT_DIR/Scripts/Phase4"
touch "$ROOT_DIR/Scripts/Phase4/cnn_basics.py"
touch "$ROOT_DIR/Scripts/Phase4/cnn_layers.py"
touch "$ROOT_DIR/Scripts/Phase4/cnn_architectures.py"
touch "$ROOT_DIR/Scripts/Phase4/rnn_basics.py"
touch "$ROOT_DIR/Scripts/Phase4/lstm_networks.py"
touch "$ROOT_DIR/Scripts/Phase4/gru_units.py"
touch "$ROOT_DIR/Scripts/Phase4/transformers_basics.py"
touch "$ROOT_DIR/Scripts/Phase4/attention_mechanisms.py"
touch "$ROOT_DIR/Scripts/Phase4/transformer_models.py"
touch "$ROOT_DIR/Scripts/Phase4/project_cifar10_train.py"

mkdir -p "$ROOT_DIR/Scripts/Phase5"
touch "$ROOT_DIR/Scripts/Phase5/autoencoders.py"
touch "$ROOT_DIR/Scripts/Phase5/vae.py"
touch "$ROOT_DIR/Scripts/Phase5/gans.py"
touch "$ROOT_DIR/Scripts/Phase5/conditional_gan.py"
touch "$ROOT_DIR/Scripts/Phase5/style_gan.py"
touch "$ROOT_DIR/Scripts/Phase5/cyclegan.py"
touch "$ROOT_DIR/Scripts/Phase5/nlp_basics.py"
touch "$ROOT_DIR/Scripts/Phase5/sequence_models.py"
touch "$ROOT_DIR/Scripts/Phase5/transformer_nlp.py"
touch "$ROOT_DIR/Scripts/Phase5/sentiment_analysis.py"
touch "$ROOT_DIR/Scripts/Phase5/project_sentiment_lstm_train.py"

mkdir -p "$ROOT_DIR/Scripts/Phase6"
touch "$ROOT_DIR/Scripts/Phase6/hyperparameter_tuning_basics.py"
touch "$ROOT_DIR/Scripts/Phase6/grid_random_search.py"
touch "$ROOT_DIR/Scripts/Phase6/bayesian_optimization.py"
touch "$ROOT_DIR/Scripts/Phase6/advanced_tuning.py"
touch "$ROOT_DIR/Scripts/Phase6/advanced_metrics.py"
touch "$ROOT_DIR/Scripts/Phase6/tf_serving_basics.py"
touch "$ROOT_DIR/Scripts/Phase6/deploy_rest_api.py"
touch "$ROOT_DIR/Scripts/Phase6/tensorflow_lite_intro.py"
touch "$ROOT_DIR/Scripts/Phase6/model_compression_quantization.py"
touch "$ROOT_DIR/Scripts/Phase6/project_deploy_rest_api.py"

mkdir -p "$ROOT_DIR/Scripts/Phase7"
touch "$ROOT_DIR/Scripts/Phase7/custom_layers.py"
touch "$ROOT_DIR/Scripts/Phase7/custom_models.py"
touch "$ROOT_DIR/Scripts/Phase7/tfx_overview.py"
touch "$ROOT_DIR/Scripts/Phase7/data_validation.py"
touch "$ROOT_DIR/Scripts/Phase7/data_transformation.py"
touch "$ROOT_DIR/Scripts/Phase7/model_training.py"
touch "$ROOT_DIR/Scripts/Phase7/model_serving.py"
touch "$ROOT_DIR/Scripts/Phase7/distributed_training.py"
touch "$ROOT_DIR/Scripts/Phase7/multi_gpu_training.py"
touch "$ROOT_DIR/Scripts/Phase7/tpu_training.py"
touch "$ROOT_DIR/Scripts/Phase7/performance_optimization.py"

mkdir -p "$ROOT_DIR/Scripts/Phase8"
touch "$ROOT_DIR/Scripts/Phase8/object_detection_basics.py"
touch "$ROOT_DIR/Scripts/Phase8/yolo_implementation.py"
touch "$ROOT_DIR/Scripts/Phase8/ssd_implementation.py"
touch "$ROOT_DIR/Scripts/Phase8/faster_rcnn_implementation.py"
touch "$ROOT_DIR/Scripts/Phase8/image_segmentation_basics.py"
touch "$ROOT_DIR/Scripts/Phase8/unet_segmentation.py"
touch "$ROOT_DIR/Scripts/Phase8/mask_rcnn.py"
touch "$ROOT_DIR/Scripts/Phase8/transfer_learning_vision.py"
touch "$ROOT_DIR/Scripts/Phase8/implement_transfer_learning.py"
touch "$ROOT_DIR/Scripts/Phase8/time_series_forecasting.py"
touch "$ROOT_DIR/Scripts/Phase8/anomaly_detection_autoencoder.py"
touch "$ROOT_DIR/Scripts/Phase8/anomaly_detection_isolation_forest.py"

mkdir -p "$ROOT_DIR/Scripts/Phase9"
touch "$ROOT_DIR/Scripts/Phase9/capstone_project_planning.py"
touch "$ROOT_DIR/Scripts/Phase9/collect_data.py"
touch "$ROOT_DIR/Scripts/Phase9/preprocess_data.py"
touch "$ROOT_DIR/Scripts/Phase9/model_development.py"
touch "$ROOT_DIR/Scripts/Phase9/model_evaluation.py"
touch "$ROOT_DIR/Scripts/Phase9/deploy_model.py"
touch "$ROOT_DIR/Scripts/Phase9/prepare_presentation.py"
touch "$ROOT_DIR/Scripts/Phase9/finalize_presentation.py"
touch "$ROOT_DIR/Scripts/Phase9/reflection.py"
touch "$ROOT_DIR/Scripts/Phase9/explore_advanced_topics.py"
touch "$ROOT_DIR/Scripts/Phase9/plan_next_steps.py"

# Global Files
touch "$ROOT_DIR/README.md"
touch "$ROOT_DIR/LICENSE"
touch "$ROOT_DIR/Requirements.txt"

echo "Directory structure created successfully."
