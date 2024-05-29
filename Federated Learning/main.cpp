#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>

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

    // Compute the weighted average of the parameters
    std::vector<torch::Tensor> avg_params;

    // Get parameters of each model
    auto params_model1 = model1.parameters();
    auto params_model2 = model2.parameters();
    auto params_model3 = model3.parameters();
    auto params_model4 = model4.parameters();
    auto base_model_params = base_model.parameters();

    // Use iterators to access parameters
    auto it_model1 = params_model1.begin();
    auto it_model2 = params_model2.begin();
    auto it_model3 = params_model3.begin();
    auto it_model4 = params_model4.begin();

    // Iterate through parameters and compute weighted average
    for (size_t i = 0; i < params_model1.size() && 
                       it_model1 != params_model1.end(); 
                       ++i, ++it_model1, ++it_model2, ++it_model3, ++it_model4) 
    {
        auto avg_param = *it_model1 * (model1_num / static_cast<float>(num_sum)) +
                         *it_model2 * (model2_num / static_cast<float>(num_sum)) +
                         *it_model3 * (model3_num / static_cast<float>(num_sum)) +
                         *it_model4 * (model4_num / static_cast<float>(num_sum));
        avg_params.push_back(avg_param);
    }

    // Set parameters of the base model to the average
    torch::NoGradGuard no_grad;
    auto it_base = base_model_params.begin();
    for (size_t i = 0; i < base_model_params.size() && it_base != base_model_params.end(); ++i, ++it_base) {
        (*it_base).copy_(avg_params[i]);
    }

    // Save the updated model
    base_model.save("model_scripted.pt");
    std::cout << "Federated Averaging completed. Model saved as model_scripted.pt" << std::endl; 
    return 0;
}
