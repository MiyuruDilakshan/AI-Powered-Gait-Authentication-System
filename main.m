clear all; close all; clc;

fprintf('========================================\n');
fprintf('GAIT AUTHENTICATION SYSTEM - MAIN SCRIPT\n');
fprintf('BINARY CLASSIFICATION APPROACH\n');
fprintf('========================================\n');

try
    %% Run Script 1: Feature Extraction 
    fprintf('\n RUNNING SCRIPT 1: FEATURE EXTRACTION\n');
    [FD_features, MD_features, raw_data] = feature_extraction(50);

    %% Run Script 2: Template Generation 
    fprintf('\n RUNNING SCRIPT 2: TEMPLATE GENERATION\n');
    templates = template_generation(FD_features, MD_features);

    %% Run Script 3: Classification & Evaluation 
    fprintf('\n RUNNING SCRIPT 3: BINARY CLASSIFICATION\n');
    results = classification(templates, raw_data);

    %% Display Summary
    fprintf('\n ALL SCRIPTS COMPLETED SUCCESSFULLY!\n');
    fprintf('========================================\n');
    fprintf('\n GENERATED OUTPUTS:\n');
    fprintf(' fig1_raw_sensor_data.png\n');
    fprintf(' fig2_binary_classifier_analysis.png\n');
    fprintf(' fig3_optimization_details.png\n');
    fprintf(' fig4_far_frr_curves.png\n');
    fprintf(' binary_classifier_results.mat\n');
    
    fprintf('\n KEY FINDINGS:\n');
    fprintf(' • Cross-Day EER: %.2f%%\n', nanmean(results.scenario2(:,3))*100);
    fprintf(' • Best Optimization: %.2f%% EER\n', min(results.optimization.overlap(:,3))*100);
    fprintf(' • Accelerometer Only: %.2f%% EER\n', nanmean(results.sensor.accelerometer(:,3))*100);
    fprintf(' • Gyroscope Only: %.2f%% EER\n', nanmean(results.sensor.gyroscope(:,3))*100);

    
catch ME
    fprintf('\n ERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
end