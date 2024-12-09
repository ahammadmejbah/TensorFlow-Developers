#!/bin/bash

# Root directory
ROOT_DIR="TensorFlow_90_Day_Guide"
mkdir -p "$ROOT_DIR"

# Function to create day directories with standard names
create_day_dir() {
    local phase_dir="$1"
    local day_num="$2"
    local day_name="$3"
    local scripts="$4"
    local additional_dirs=("${!5}")

    mkdir -p "$phase_dir/Day${day_num}_${day_name}/Scripts"
    
    # Create placeholder files
    touch "$phase_dir/Day${day_num}_${day_name}/README.md"
    touch "$phase_dir/Day${day_num}_${day_name}/Lecture_Notes.ipynb"
    touch "$phase_dir/Day${day_num}_${day_name}/Exercises.ipynb"
    touch "$phase_dir/Day${day_num}_${day_name}/Resources.md"
    
    # Create script files
    for script in "${scripts[@]}"; do
        touch "$phase_dir/Day${day_num}_${day_name}/Scripts/$script"
    done

    # Create additional directories if any
    for add_dir in "${additional_dirs[@]}"; do
        mkdir -p "$phase_dir/Day${day_num}_${day_name}/$add_dir"
    done
}

# Phase Definitions
declare -A phases
phases=(
    ["Phase1_Foundations_of_Python_and_Mathematics"]="Phase1_Foundations_of_Python_and_Mathematics"
    ["Phase2_Introduction_to_TensorFlow_and_Keras"]="Phase2_Introduction_to_TensorFlow_and_Keras"
    ["Phase3_Core_Concepts_of_Neural_Networks"]="Phase3_Core_Concepts_of_Neural_Networks"
    ["Phase4_Advanced_Neural_Network_Architectures"]="Phase4_Advanced_Neural_Network_Architectures"
    ["Phase5_Specialized_Models_and_Techniques"]="Phase5_Specialized_Models_and_Techniques"
    ["Phase6_Model_Optimization_and_Deployment"]="Phase6_Model_Optimization_and_Deployment"
    ["Phase7_Advanced_TensorFlow_Techniques_and_Applications"]="Phase7_Advanced_TensorFlow_Techniques_and_Applications"
    ["Phase8_Exploring_Various_Applications"]="Phase8_Exploring_Various_Applications"
    ["Phase9_Capstone_Projects_and_Review"]="Phase9_Capstone_Projects_and_Review"
)

# Create Phases
for phase in "${!phases[@]}"; do
    mkdir -p "$ROOT_DIR/${phases[$phase]}"
done

# Phase 1: Days 1-10
declare -a phase1_days=(
    "Python_Basics_Refresher:basics.py"
    "Python_Data_Structures:data_structures.py"
    "Functions_and_Modules:functions_modules.py"
    "Object_Oriented_Programming:oop.py"
    "Introduction_to_NumPy:numpy_basics.py"
    "Data_Manipulation_with_Pandas:pandas_manipulation.py"
    "Data_Visualization_with_Matplotlib_and_Seaborn:data_visualization.py"
    "Linear_Algebra_Fundamentals:linear_algebra.py"
    "Calculus_for_Machine_Learning:calculus_ml.py"
    "Probability_and_Statistics_Basics:probability_statistics.py"
)

phase_dir="$ROOT_DIR/Phase1_Foundations_of_Python_and_Mathematics"

for day in "${phase1_days[@]}"; do
    IFS=":" read -r day_name script <<< "$day"
    create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=()"
done

# Phase 2: Days 11-20
declare -a phase2_days=(
    "Setting_Up_the_Environment:install_dependencies.py"
    "TensorFlow_Basics:tensorflow_basics.py"
    "Keras_API_Overview:keras_api_overview.py"
    "Building_Your_First_Neural_Network:first_neural_network.py"
    "Compiling_and_Training_Models:compile_train_models.py"
    "Evaluating_and_Improving_Models:evaluate_improve_models.py"
    "Data_Handling_with_TensorFlow:data_handling_tf.py"
    "Working_with_Datasets:working_with_datasets.py"
    "Saving_and_Loading_Models:saving_loading_models.py"
    "Mini_Project_Handwritten_Digit_Classification:mini_project_train.py"
)

phase_dir="$ROOT_DIR/Phase2_Introduction_to_TensorFlow_and_Keras"

for day in "${phase2_days[@]}"; do
    IFS=":" read -r day_name script <<< "$day"
    if [[ "$day_name" == "Mini_Project_Handwritten_Digit_Classification" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=('Data/MNIST' 'Notebooks' 'Models/saved_model' 'Results')"
    else
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=()"
    fi
done

# Phase 3: Days 21-30
declare -a phase3_days=(
    "Understanding_Perceptrons_and_Neurons:perceptron.py"
    "Activation_Functions_in_Depth:activation_functions.py"
    "Loss_Functions_Explained:loss_functions.py"
    "Optimization_Algorithms:optimization_algorithms.py"
    "Regularization_Techniques:regularization_techniques.py"
    "Batch_Normalization_and_Layer_Normalization:normalization.py"
    "Introduction_to_Dropout:dropout.py"
    "Review_and_Implement_a_Feedforward_Neural_Network:feedforward_nn.py"
    "Introduction_to_Model_Evaluation_Metrics:evaluation_metrics.py"
    "Hands-On_with_Evaluation_Metrics:apply_evaluation_metrics.py"
)

phase_dir="$ROOT_DIR/Phase3_Core_Concepts_of_Neural_Networks"

for day in "${phase3_days[@]}"; do
    IFS=":" read -r day_name script <<< "$day"
    if [[ "$day_name" == "Review_and_Implement_a_Feedforward_Neural_Network" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=('Implementation')"
    else
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=()"
    fi
done

# Phase 4: Days 31-40
declare -a phase4_days=(
    "Introduction_to_Convolutional_Neural_Networks:cnn_basics.py"
    "Deep_Dive_into_CNN_Layers:cnn_layers.py"
    "Architectures_of_CNNs:cnn_architectures.py"
    "Introduction_to_Recurrent_Neural_Networks:rnn_basics.py"
    "Long_Short-Term_Memory_Networks:lstm_networks.py"
    "Gated_Recurrent_Units:gru_units.py"
    "Introduction_to_Transformers_and_Attention_Mechanisms:transformers_basics.py"
    "Implementing_Attention_Mechanisms:attention_mechanisms.py"
    "Transformer_Models_for_NLP:transformer_models.py"
    "Project_Image_Classification_with_CNNs_on_CIFAR-10:project_cifar10_train.py"
)

phase_dir="$ROOT_DIR/Phase4_Advanced_Neural_Network_Architectures"

for day in "${phase4_days[@]}"; do
    IFS=":" read -r day_name script <<< "$day"
    if [[ "$day_name" == "Project_Image_Classification_with_CNNs_on_CIFAR-10" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=('Data/CIFAR-10' 'Notebooks/CIFAR10_Image_Classification.ipynb' 'Models/saved_model' 'Results')"
    else
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=()"
    fi
done

# Phase 5: Days 41-50
declare -a phase5_days=(
    "Introduction_to_Autoencoders:autoencoders.py"
    "Variational_Autoencoders:vae.py"
    "Generative_Adversarial_Networks:gans.py"
    "Advanced_GAN_Techniques:conditional_gan.py style_gan.py cyclegan.py"
    "Introduction_to_NLP_with_TensorFlow:nlp_basics.py"
    "Sequence_Models_for_NLP:sequence_models.py"
    "Transformer_Models_for_NLP:transformer_nlp.py"
    "Text_Classification_and_Sentiment_Analysis:sentiment_analysis.py"
    "Project_Sentiment_Analysis_Using_LSTM:project_sentiment_lstm_train.py"
    "Review_and_Consolidation:review_consolidation.py"
)

phase_dir="$ROOT_DIR/Phase5_Specialized_Models_and_Techniques"

for day in "${phase5_days[@]}"; do
    IFS=":" read -r day_name scripts <<< "$day"
    IFS=' ' read -ra script_array <<< "$scripts"
    if [[ "$day_name" == "Project_Sentiment_Analysis_Using_LSTM" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=('Data/IMDb' 'Notebooks/Sentiment_Analysis_LSTM.ipynb' 'Models/saved_model' 'Results')"
    elif [[ "$day_name" == "Advanced_GAN_Techniques" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=()"
    else
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=()"
    fi
done

# Phase 6: Days 51-60
declare -a phase6_days=(
    "Hyperparameter_Tuning_Basics:hyperparameter_tuning_basics.py"
    "Grid_Search_and_Random_Search:grid_random_search.py"
    "Bayesian_Optimization_for_Hyperparameter_Tuning:bayesian_optimization.py"
    "Advanced_Hyperparameter_Tuning_Techniques:advanced_tuning.py"
    "Model_Evaluation_Metrics_Revisited:advanced_metrics.py"
    "TensorFlow_Serving_and_Model_Deployment_Basics:tf_serving_basics.py"
    "Deploying_Models_as_REST_APIs:deploy_rest_api.py"
    "Introduction_to_TensorFlow_Lite:tensorflow_lite_intro.py"
    "Model_Compression_and_Quantization:model_compression_quantization.py"
    "Project_Deploy_a_Trained_Model_as_a_REST_API:project_deploy_rest_api.py"
)

phase_dir="$ROOT_DIR/Phase6_Model_Optimization_and_Deployment"

for day in "${phase6_days[@]}"; do
    IFS=":" read -r day_name script <<< "$day"
    if [[ "$day_name" == "Project_Deploy_a_Trained_Model_as_a_REST_API" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=('Data/CIFAR-10' 'Notebooks/Deploy_CIFAR10_CNN_REST_API.ipynb' 'Models/saved_model' 'API' 'Results')"
    else
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "$script" "additional_dirs=()"
    fi
done

# Phase 7: Days 61-70
declare -a phase7_days=(
    "Custom_Layers_in_Keras:custom_layers.py"
    "Custom_Models_in_Keras:custom_models.py"
    "TensorFlow_Extended_TFX_Overview:tfx_overview.py"
    "Data_Validation_and_Transformation_with_TFX:data_validation.py data_transformation.py"
    "Model_Training_and_Serving_with_TFX:model_training.py model_serving.py"
    "Distributed_Training_with_TensorFlow:distributed_training.py"
    "Multi-GPU_Training_Strategies:multi_gpu_training.py"
    "TPU_Training_with_TensorFlow:tpu_training.py"
    "Optimizing_TensorFlow_Performance:performance_optimization.py"
    "Review_and_Practice_with_Advanced_TensorFlow_Features:advanced_features_practice.py"
)

phase_dir="$ROOT_DIR/Phase7_Advanced_TensorFlow_Techniques_and_Applications"

for day in "${phase7_days[@]}"; do
    IFS=":" read -r day_name scripts <<< "$day"
    IFS=' ' read -ra script_array <<< "$scripts"
    create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=()"
done

# Phase 8: Days 71-80
declare -a phase8_days=(
    "Introduction_to_Object_Detection:object_detection_basics.py"
    "Implementing_YOLO_with_TensorFlow:yolo_implementation.py"
    "Implementing_SSD_and_Faster_R-CNN:ssd_implementation.py faster_rcnn_implementation.py"
    "Image_Segmentation_Basics:image_segmentation_basics.py"
    "Implementing_U-Net_for_Image_Segmentation:unet_segmentation.py"
    "Implementing_Mask_R-CNN:mask_rcnn.py"
    "Transfer_Learning_for_Vision_Tasks:transfer_learning_vision.py"
    "Implementing_Transfer_Learning:implement_transfer_learning.py"
    "Introduction_to_Time_Series_Forecasting:time_series_forecasting.py"
    "Anomaly_Detection_Techniques:anomaly_detection_autoencoder.py anomaly_detection_isolation_forest.py"
)

phase_dir="$ROOT_DIR/Phase8_Exploring_Various_Applications"

for day in "${phase8_days[@]}"; do
    IFS=":" read -r day_name scripts <<< "$day"
    IFS=' ' read -ra script_array <<< "$scripts"
    if [[ "$day_name" == "Implementing_YOLO_with_TensorFlow" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=('Scripts')"
    elif [[ "$day_name" == "Implementing_SSD_and_Faster_R-CNN" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=()"
    elif [[ "$day_name" == "Anomaly_Detection_Techniques" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=()"
    elif [[ "$day_name" == "Project_*" ]]; then
        # Handle project-specific directories if any
        :
    else
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=()"
    fi
done

# Phase 9: Days 81-90
declare -a phase9_days=(
    "Capstone_Project_Planning:capstone_project_planning.py"
    "Data_Collection_and_Preprocessing:collect_data.py preprocess_data.py"
    "Model_Development_and_Training:model_development.py"
    "Model_Evaluation_and_Optimization:model_evaluation.py"
    "Model_Deployment_and_Documentation:deploy_model.py"
    "Presentation_Preparation:prepare_presentation.py"
    "Finalize_and_Present_the_Capstone_Project:finalize_presentation.py"
    "Review_and_Reflect_on_Learning_Journey:reflection.py"
    "Explore_Advanced_Topics_or_Certifications:explore_advanced_topics.py"
    "Plan_Next_Steps_and_Continued_Learning:plan_next_steps.py"
)

phase_dir="$ROOT_DIR/Phase9_Capstone_Projects_and_Review"

for day in "${phase9_days[@]}"; do
    IFS=":" read -r day_name scripts <<< "$day"
    IFS=' ' read -ra script_array <<< "$scripts"
    if [[ "$day_name" == "Capstone_Project_Planning" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=('Project_Selection_Guide' 'Project_Ideas')"
    elif [[ "$day_name" == "Model_Deployment_and_Documentation" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=('Deployment_Scripts' 'Documentation')"
    elif [[ "$day_name" == "Finalize_and_Present_the_Capstone_Project" ]]; then
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=('Final_Presentation' 'Feedback')"
    else
        create_day_dir "$phase_dir" "${day_name%%_*}" "$day_name" "${script_array[@]}" "additional_dirs=()"
    fi
done

# Additional Tips and Resources
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

# Projects Directory
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
# Create placeholder files for IMDb data
touch "$ROOT_DIR/Data/IMDb/aclImdb/train/"*
touch "$ROOT_DIR/Data/IMDb/aclImdb/test/"*
touch "$ROOT_DIR/Data/IMDb/aclImdb/unsup/"*

mkdir -p "$ROOT_DIR/Data/custom_dataset/images"
mkdir -p "$ROOT_DIR/Data/custom_dataset/annotations"
# Create placeholder files for custom dataset
touch "$ROOT_DIR/Data/custom_dataset/images/"*
touch "$ROOT_DIR/Data/custom_dataset/annotations/"*

# Models Directory
mkdir -p "$ROOT_DIR/Models/saved_model"
mkdir -p "$ROOT_DIR/Models/saved_transformer_model"
mkdir -p "$ROOT_DIR/Models/saved_object_detector"
mkdir -p "$ROOT_DIR/Models/custom_models"
# Create placeholder files for Models
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

# Phase1 Scripts
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

# Phase2 Scripts
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

# Phase3 Scripts
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

# Phase4 Scripts
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

# Phase5 Scripts
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

# Phase6 Scripts
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

# Phase7 Scripts
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

# Phase8 Scripts
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

# Phase9 Scripts
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

echo "TensorFlow 90-Day Guide directory structure created successfully!"
