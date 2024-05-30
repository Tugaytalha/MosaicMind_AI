#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>
#include<ctime>
float beforeCompiling(const std::vector<torch::Tensor>& params)
{
    std::string file_name = "before_compile";
    std::time_t now = std::time(0);
    std::tm* now_tm = std::localtime(&now);

    // Create a time suffix for the file name to ensure it's unique
    std::ostringstream time_suffix;
    time_suffix << std::setfill('0') << std::setw(2) << now_tm->tm_hour
                << std::setw(2) << now_tm->tm_min
                << std::setw(2) << now_tm->tm_sec; // Added seconds for extra uniqueness

    file_name += "_" + time_suffix.str() + ".txt"; // Append time suffix to file name
    std::ofstream file(file_name);
    double sum_abs = 0.0;
    int numOfElements =0;

    for(const auto& param: params)
    {
        sum_abs+= param.abs().sum().item<double>();
        numOfElements += param.numel();
    }
    std::cout << "başlamadan önce weightlerin ortalaması " << sum_abs/ numOfElements << std::endl;
    file << "res=" << sum_abs / numOfElements << std::endl;
    file.close();
    return sum_abs / numOfElements;
}
float compute_mean_abs_params(const std::vector<torch::Tensor>& params) {
    double sum_abs = 0.0;
    int64_t total_elements = 0;
    for (const auto& param : params) {
        sum_abs += (float) param.abs().sum().item<double>();
        total_elements += param.numel();
    }
    std::cout << " sum\'a bakalım mantıklı bir sayı mı " << sum_abs << std::endl;
    std:: cout << "total element in the model  "<< total_elements << std::endl;
    return total_elements > 0 ? sum_abs / total_elements : 0.0;
}

// Function to save model parameters to a file
void save_model_params_to_file(const std::vector<torch::Tensor>& params, std::string file_name, int max_lines = 10) {
    std::time_t now = std::time(0);
    std::tm* now_tm = std::localtime(&now);

    // Create a time suffix for the file name to ensure it's unique
    std::ostringstream time_suffix;
    time_suffix << std::setfill('0') << std::setw(2) << now_tm->tm_hour
                << std::setw(2) << now_tm->tm_min
                << std::setw(2) << now_tm->tm_sec; // Added seconds for extra uniqueness

    file_name += "_" + time_suffix.str() + ".txt"; // Append time suffix to file name

    std::cout << "Creating file: " << file_name << " for debug purposes..." << std::endl;

    std::ofstream file(file_name);
    if (!file) {
        std::cerr << "Unable to open file: " << file_name << std::endl;
        return;
    }

    int lines_written = 0;
    for (const auto& param : params) {
        if (lines_written >= max_lines) break;

        auto param_data = param.data_ptr<float>();
        for (int64_t i = 0; i < param.numel() && lines_written < max_lines; ++i) {
            file << param_data[i] << "\n";
            ++lines_written;
        }
    }
    file << "Mean Absolute Value of Parameters: ";
    file << compute_mean_abs_params(params) << "\n";
    file.close();
}
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./federated_averaging <json_file_path>" << std::endl;
        return 1;
    }

    std::string json_file_path = argv[1];

    // Load and parse JSON file
    std::ifstream json_file(json_file_path);
    if (!json_file) {
        std::cerr << "Unable to open JSON file: " << json_file_path << std::endl;
        return 1;
    }

    Json::Value json_data;
    json_file >> json_data;

    // Load the models
    torch::jit::script::Module model1 = torch::jit::load(json_data["model1"].asString());
    torch::jit::script::Module model2 = torch::jit::load(json_data["model2"].asString());
    torch::jit::script::Module model3 = torch::jit::load(json_data["model3"].asString());
    torch::jit::script::Module model4 = torch::jit::load(json_data["model4"].asString());
    torch::jit::script::Module base_model = torch::jit::load(json_data["base_model"].asString());

    // Extract model counts from JSON
    int64_t model1_num = json_data["model1_num"].asInt64();
    int64_t model2_num = json_data["model2_num"].asInt64();
    int64_t model3_num = json_data["model3_num"].asInt64();
    int64_t model4_num = json_data["model4_num"].asInt64();

    int64_t num_sum = model1_num + model2_num + model3_num + model4_num;

    // Convert parameters to vectors of tensors
    auto get_parameters = [](torch::jit::script::Module& module) {
        std::vector<torch::Tensor> params;
        for (const auto& param : module.parameters()) {
            params.push_back(param);
        }
        return params;
    };

    std::vector<torch::Tensor> params_model1 = get_parameters(model1);
    std::vector<torch::Tensor> params_model2 = get_parameters(model2);
    std::vector<torch::Tensor> params_model3 = get_parameters(model3);
    std::vector<torch::Tensor> params_model4 = get_parameters(model4);
    std::vector<torch::Tensor> base_model_params = get_parameters(base_model);
    beforeCompiling(base_model_params);

    // Compute the weighted average of the parameters
    std::vector<torch::Tensor> avg_params;
    for (size_t i = 0; i < params_model1.size(); ++i) {
        auto avg_param = params_model1[i] * (model1_num / static_cast<float>(num_sum)) +
                         params_model2[i] * (model2_num / static_cast<float>(num_sum)) +
                         params_model3[i] * (model3_num / static_cast<float>(num_sum)) +
                         params_model4[i] * (model4_num / static_cast<float>(num_sum));
        avg_params.push_back(avg_param);
    }

    // Set parameters of the base model to the average
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < base_model_params.size(); ++i) {
        base_model_params[i].copy_(avg_params[i]);
    }

    // Save the updated model
    base_model.save("model_scripted.pt");

    // Save model parameters to files
    save_model_params_to_file(params_model1, "model1_params.txt");
    save_model_params_to_file(params_model2, "model2_params.txt");
    save_model_params_to_file(params_model3, "model3_params.txt");
    save_model_params_to_file(params_model4, "model4_params.txt");
    save_model_params_to_file(avg_params, "base_model_params.txt");

    std::cout << "Federated Averaging completed. Model saved as model_scripted.pt" << std::endl; 
    return 0;
}