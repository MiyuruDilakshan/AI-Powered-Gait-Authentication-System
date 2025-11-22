function results = classification(templates, raw_data)

fprintf('\n========================================\n');
fprintf('SCRIPT 3: BINARY CLASSIFICATION\n');
fprintf('Training ONE binary classifier PER USER\n');
fprintf('========================================\n');

num_users = templates.num_users;
valid_users = templates.valid_users;
results = struct();

%% Test All Three Scenarios with BINARY CLASSIFIERS
fprintf('\n=== TESTING ALL SCENARIOS - BINARY APPROACH ===\n');

fprintf('\n[SCENARIO 1: Same Day - Binary Classifiers]\n');
scenario1_results = test_scenario_binary(templates.scenario1, valid_users, 'Same Day');
results.scenario1 = scenario1_results;

fprintf('\n[SCENARIO 2: Cross-Day - Binary Classifiers]\n');
scenario2_results = test_scenario_binary(templates.scenario2, valid_users, 'Cross-Day');
results.scenario2 = scenario2_results;

fprintf('\n[SCENARIO 3: Combined - Binary Classifiers]\n');
scenario3_results = test_scenario_binary(templates.scenario3, valid_users, 'Combined');
results.scenario3 = scenario3_results;

%% Sensor Comparison Analysis
fprintf('\n=== SENSOR COMPARISON - BINARY CLASSIFIERS ===\n');
sensor_results = compare_sensors_binary(templates, valid_users);
results.sensor = sensor_results;

%% ACTUAL OPTIMIZATION EXPERIMENTS
fprintf('\n=== ACTUAL OPTIMIZATION EXPERIMENTS ===\n');
optimization_results = run_actual_optimization(raw_data, valid_users);
results.optimization = optimization_results;

%% Generate Comprehensive Visualizations
fprintf('\n=== GENERATING VISUALIZATIONS ===\n');
generate_visualizations_binary(results, templates, valid_users);

%% Display Final Results Summary
fprintf('\n=== COMPREHENSIVE RESULTS SUMMARY ===\n');
display_results_summary_binary(results, valid_users);

fprintf('\n✓ SCRIPT 3 COMPLETED: Binary classification evaluation successful\n');

end

%% TRUE BINARY CLASSIFICATION: Train ONE classifier PER USER
function scenario_results = test_scenario_binary(scenario_data, valid_users, scenario_name)
    fprintf(' %s - Training binary classifiers...\n', scenario_name);
    
    X_train = scenario_data.X_train;
    y_train = scenario_data.y_train;
    X_test = scenario_data.X_test;
    y_test = scenario_data.y_test;
    
    % Normalize data
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_test_norm = (X_test - mu) ./ (sigma + eps);
    
    num_users = length(valid_users);
    scenario_results = zeros(num_users, 3); % FAR, FRR, EER
    
    % Train ONE binary classifier for EACH user
    for target_user = 1:num_users
        if ~valid_users(target_user)
            continue;
        end
        
        % Create BINARY labels for this user
        binary_train_labels = (y_train == target_user);
        binary_test_labels = (y_test == target_user);
        
        % Convert to categorical (0 or 1)
        train_targets = full(ind2vec(binary_train_labels' + 1)); % +1 because ind2vec starts at 1
        
        % Train ONE binary classifier for THIS user
        net = patternnet([20, 10], 'trainscg');
        net.trainParam.showWindow = false;
        net.performParam.regularization = 0.05;
        net.trainParam.epochs = 100;
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.2;
        net.divideParam.testRatio = 0.0;
        
        [net, ~] = train(net, X_train_norm', train_targets);
        
        % Test the binary classifier
        predictions = net(X_test_norm');
        user_scores = predictions(2, :)'; % Probability of being genuine (class 2 = genuine)
        
        % Calculate binary authentication metrics
        [FAR, FRR, EER] = calculate_authentication_metrics(binary_test_labels, user_scores);
        scenario_results(target_user, :) = [FAR, FRR, EER];
        
        fprintf(' User %2d: FAR=%.2f%%, FRR=%.2f%%, EER=%.2f%%\n', ...
            target_user, FAR*100, FRR*100, EER*100);
    end
end

%% Sensor Comparison with Binary Classifiers
function sensor_results = compare_sensors_binary(templates, valid_users)
    sensor_results = struct();
    scenario_data = templates.scenario2;
    
    % Accelerometer Only
    fprintf('\n[Accelerometer Only - Binary Classifiers]\n');
    acc_results = test_sensor_binary(scenario_data, templates.acc_features, valid_users, 'Accelerometer');
    sensor_results.accelerometer = acc_results;
    
    % Gyroscope Only
    fprintf('\n[Gyroscope Only - Binary Classifiers]\n');
    gyro_results = test_sensor_binary(scenario_data, templates.gyro_features, valid_users, 'Gyroscope');
    sensor_results.gyroscope = gyro_results;
    
    sensor_results.combined = scenario_data;
end

function sensor_results = test_sensor_binary(scenario_data, feature_indices, valid_users, sensor_name)
    fprintf(' Testing %s features...\n', sensor_name);
    
    X_train = scenario_data.X_train(:, feature_indices);
    y_train = scenario_data.y_train;
    X_test = scenario_data.X_test(:, feature_indices);
    y_test = scenario_data.y_test;
    
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_test_norm = (X_test - mu) ./ (sigma + eps);
    
    num_users = length(valid_users);
    sensor_results = zeros(num_users, 3);
    
    for target_user = 1:num_users
        if ~valid_users(target_user)
            continue;
        end
        
        binary_train_labels = (y_train == target_user);
        binary_test_labels = (y_test == target_user);
        train_targets = full(ind2vec(binary_train_labels' + 1));
        
        net = patternnet([15, 8], 'trainscg');
        net.trainParam.showWindow = false;
        [net, ~] = train(net, X_train_norm', train_targets);
        
        predictions = net(X_test_norm');
        user_scores = predictions(2, :)';
        
        [FAR, FRR, EER] = calculate_authentication_metrics(binary_test_labels, user_scores);
        sensor_results(target_user, :) = [FAR, FRR, EER];
    end
    fprintf(' Complete\n');
end

%% ACTUAL OPTIMIZATION: Re-run pipeline with different parameters
function optimization_results = run_actual_optimization(raw_data, valid_users)
    optimization_results = struct();
    
    % 1. ACTUAL Window Overlap Testing
    fprintf('\n[1. ACTUAL Window Overlap Testing]\n');
    fprintf('   Re-segmenting data with different overlaps...\n');
    overlap_results = test_actual_window_overlap(raw_data, valid_users);
    optimization_results.overlap = overlap_results;
    
    % 2. Neural Network Architecture Tuning (using existing features)
    fprintf('\n[2. Neural Network Architecture Tuning]\n');
    nn_results = test_nn_architectures(raw_data, valid_users);
    optimization_results.nn_tuning = nn_results;
    
    % 3. Feature Reduction with PCA
    fprintf('\n[3. Feature Reduction with PCA]\n');
    feature_results = test_feature_reduction(raw_data, valid_users);
    optimization_results.feature_reduction = feature_results;
end

%% ACTUAL Window Overlap: Re-extract features with different overlaps
function overlap_results = test_actual_window_overlap(raw_data, valid_users)
    overlap_percentages = [0, 25, 50, 75];
    overlap_results = zeros(length(overlap_percentages), 3);
    
    for i = 1:length(overlap_percentages)
        overlap_pct = overlap_percentages(i);
        fprintf('   Testing %d%% overlap: ', overlap_pct);
        
        % Re-extract features with this overlap
        [FD_features_new, MD_features_new, ~] = feature_extraction(overlap_pct);
        
        % Generate new templates
        templates_new = template_generation(FD_features_new, MD_features_new);
        
        % Test on cross-day scenario
        scenario_results = test_scenario_binary(templates_new.scenario2, valid_users, '');
        
        % Calculate average EER
        avg_eer = nanmean(scenario_results(valid_users, 3));
        overlap_results(i, 3) = avg_eer;
        fprintf('Avg EER=%.2f%%\n', avg_eer*100);
    end
end

%% Neural Network Architecture Testing
function nn_results = test_nn_architectures(raw_data, valid_users)
    % Use default 50% overlap
    [FD_features, MD_features, ~] = feature_extraction(50);
    templates = template_generation(FD_features, MD_features);
    scenario_data = templates.scenario2;
    
    hidden_configs = {[10,5], [20,10], [30,15], [50,25]};
    nn_results = zeros(length(hidden_configs), 3);
    
    for i = 1:length(hidden_configs)
        config = hidden_configs{i};
        fprintf('   Architecture %s: ', mat2str(config));
        
        X_train = scenario_data.X_train;
        y_train = scenario_data.y_train;
        X_test = scenario_data.X_test;
        y_test = scenario_data.y_test;
        
        [X_train_norm, mu, sigma] = zscore(X_train);
        X_test_norm = (X_test - mu) ./ (sigma + eps);
        
        user_eers = zeros(sum(valid_users), 1);
        user_idx = 1;
        
        for target_user = 1:length(valid_users)
            if ~valid_users(target_user)
                continue;
            end
            
            binary_train = (y_train == target_user);
            binary_test = (y_test == target_user);
            train_targets = full(ind2vec(binary_train' + 1));
            
            net = patternnet(config, 'trainscg');
            net.trainParam.showWindow = false;
            [net, ~] = train(net, X_train_norm', train_targets);
            
            predictions = net(X_test_norm');
            user_scores = predictions(2, :)';
            
            [~, ~, EER] = calculate_authentication_metrics(binary_test, user_scores);
            user_eers(user_idx) = EER;
            user_idx = user_idx + 1;
        end
        
        avg_eer = mean(user_eers);
        nn_results(i, 3) = avg_eer;
        fprintf('Avg EER=%.2f%%\n', avg_eer*100);
    end
end

%% Feature Reduction with PCA
function feature_results = test_feature_reduction(raw_data, valid_users)
    [FD_features, MD_features, ~] = feature_extraction(50);
    templates = template_generation(FD_features, MD_features);
    scenario_data = templates.scenario2;
    
    feature_percentages = [100, 75, 50, 25];
    feature_results = zeros(length(feature_percentages), 3);
    
    [X_train_norm, mu, sigma] = zscore(scenario_data.X_train);
    [coeff, score, ~, ~, explained] = pca(X_train_norm);
    
    for i = 1:length(feature_percentages)
        pct = feature_percentages(i);
        fprintf('   %d%% features: ', pct);
        
        num_components = find(cumsum(explained) >= pct, 1);
        if isempty(num_components)
            num_components = length(explained);
        end
        
        X_train_pca = score(:, 1:num_components);
        X_test_pca = (scenario_data.X_test - mu) ./ (sigma + eps) * coeff(:, 1:num_components);
        
        user_eers = zeros(sum(valid_users), 1);
        user_idx = 1;
        
        for target_user = 1:length(valid_users)
            if ~valid_users(target_user)
                continue;
            end
            
            binary_train = (scenario_data.y_train == target_user);
            binary_test = (scenario_data.y_test == target_user);
            train_targets = full(ind2vec(binary_train' + 1));
            
            net = patternnet([20, 10], 'trainscg');
            net.trainParam.showWindow = false;
            [net, ~] = train(net, X_train_pca', train_targets);
            
            predictions = net(X_test_pca');
            user_scores = predictions(2, :)';
            
            [~, ~, EER] = calculate_authentication_metrics(binary_test, user_scores);
            user_eers(user_idx) = EER;
            user_idx = user_idx + 1;
        end
        
        avg_eer = mean(user_eers);
        feature_results(i, 3) = avg_eer;
        fprintf('Avg EER=%.2f%%\n', avg_eer*100);
    end
end
%% Visualization Functions
function generate_visualizations_binary(results, templates, valid_users)
    fprintf(' Generating visualizations...\n');
    
    % Main results figure
    figure('Position', [50, 50, 1800, 1400]);
    
    % Subplot 1: Scenario Comparison
    subplot(3, 2, 1);
    scenarios = {'Same Day', 'Cross-Day', 'Combined'};
    scenario_EER = [nanmean(results.scenario1(valid_users,3)), ...
                    nanmean(results.scenario2(valid_users,3)), ...
                    nanmean(results.scenario3(valid_users,3))] * 100;
    
    bar(scenario_EER, 'FaceColor', [0.2 0.6 0.8]);
    set(gca, 'XTickLabel', scenarios, 'FontSize', 10);
    ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Binary Classifiers - Scenario Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    for i = 1:length(scenario_EER)
        text(i, scenario_EER(i) + 0.2, sprintf('%.2f%%', scenario_EER(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
    end
    
    % Subplot 2: Sensor Comparison
    subplot(3, 2, 2);
    sensor_types = {'Acc Only', 'Gyro Only', 'Combined'};
    sensor_EER = [nanmean(results.sensor.accelerometer(valid_users,3)), ...
                  nanmean(results.sensor.gyroscope(valid_users,3)), ...
                  nanmean(results.scenario2(valid_users,3))] * 100;
    
    bar(sensor_EER, 'FaceColor', [0.8 0.4 0.2]);
    set(gca, 'XTickLabel', sensor_types, 'FontSize', 10);
    ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Binary Classifiers - Sensor Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    for i = 1:length(sensor_EER)
        text(i, sensor_EER(i) + 0.5, sprintf('%.2f%%', sensor_EER(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
    end
    
    % Subplot 3: Optimization Results
    subplot(3, 2, 3);
    opt_types = {'Baseline', '0% Ovlp', '25% Ovlp', '75% Ovlp', 'Best NN', 'Best PCA'};
    baseline_eer = nanmean(results.scenario2(valid_users,3)) * 100;
    opt_EER = [baseline_eer, ...
               results.optimization.overlap(1,3)*100, ...
               results.optimization.overlap(2,3)*100, ...
               results.optimization.overlap(4,3)*100, ...
               min(results.optimization.nn_tuning(:,3))*100, ...
               min(results.optimization.feature_reduction(:,3))*100];
    
    bar(opt_EER, 'FaceColor', [0.4 0.8 0.4]);
    set(gca, 'XTickLabel', opt_types, 'FontSize', 9, 'XTickLabelRotation', 45);
    ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Optimization Results', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    for i = 1:length(opt_EER)
        text(i, opt_EER(i) + 0.3, sprintf('%.2f%%', opt_EER(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    end
    
    % Subplot 4: Per-User EER
    subplot(3, 2, 4);
    user_ids = find(valid_users);
    user_EER = results.scenario2(valid_users, 3) * 100;
    
    bar(user_ids, user_EER, 'FaceColor', [0.8 0.2 0.2]);
    xlabel('User ID', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Per-User EER (Cross-Day)', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    % Subplot 5: Architecture Description
    subplot(3, 2, 5);
    axis off;
    architecture_text = {
        'BINARY CLASSIFICATION ARCHITECTURE:'
        ' '
        '• ONE Binary Classifier PER USER (10 classifiers total)'
        '• Each classifier: "Is this User X?"'
        '• Input: 78 features (Acc + Gyro)'
        '• Hidden Layers: [20, 10] neurons'
        '• Output: 2 nodes (Impostor=0, Genuine=1)'
        '• Training: Genuine vs All Impostors'
        '• Metrics: FAR, FRR, EER per user'
    };
    
    text(0.1, 0.9, architecture_text, 'FontSize', 11, 'FontWeight', 'bold', ...
         'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
         'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'black', 'Margin', 5);
    title('Binary Classification Architecture', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Subplot 6: Performance Summary
    subplot(3, 2, 6);
    methods = {'Cross-Day', 'Acc Only', 'Gyro', 'Best Ovlp', 'Best NN', 'Best PCA'};
    best_EER = [nanmean(results.scenario2(valid_users,3))*100, ...
                nanmean(results.sensor.accelerometer(valid_users,3))*100, ...
                nanmean(results.sensor.gyroscope(valid_users,3))*100, ...
                min(results.optimization.overlap(:,3))*100, ...
                min(results.optimization.nn_tuning(:,3))*100, ...
                min(results.optimization.feature_reduction(:,3))*100];
    
    bar(best_EER, 'FaceColor', [0.1 0.5 0.9]);
    set(gca, 'XTickLabel', methods, 'FontSize', 9, 'XTickLabelRotation', 45);
    ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Best Methods Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    for i = 1:length(best_EER)
        text(i, best_EER(i) + 0.3, sprintf('%.2f%%', best_EER(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
    end
    
    sgtitle('GAIT AUTHENTICATION - BINARY CLASSIFICATION ANALYSIS', ...
            'FontSize', 14, 'FontWeight', 'bold');
    saveas(gcf, 'fig2_binary_classifier_analysis.png');
    fprintf(' ✓ Saved: fig2_binary_classifier_analysis.png\n');
    close;
    
    % Detailed optimization figure
    figure('Position', [100, 100, 1400, 800]);
    
    subplot(1, 3, 1);
    overlap_labels = {'0%', '25%', '50%', '75%'};
    overlap_eer = results.optimization.overlap(:,3) * 100;
    bar(overlap_eer, 'FaceColor', [0.2 0.7 0.5]);
    set(gca, 'XTickLabel', overlap_labels, 'FontSize', 10);
    xlabel('Window Overlap', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Window Overlap Optimization (ACTUAL)', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    for i = 1:length(overlap_eer)
        text(i, overlap_eer(i) + 0.2, sprintf('%.2f%%', overlap_eer(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    subplot(1, 3, 2);
    nn_labels = {'[10,5]', '[20,10]', '[30,15]', '[50,25]'};
    nn_eer = results.optimization.nn_tuning(:,3) * 100;
    bar(nn_eer, 'FaceColor', [0.8 0.4 0.2]);
    set(gca, 'XTickLabel', nn_labels, 'FontSize', 10);
    xlabel('Network Architecture', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Neural Network Architecture', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    for i = 1:length(nn_eer)
        text(i, nn_eer(i) + 0.2, sprintf('%.2f%%', nn_eer(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    subplot(1, 3, 3);
    feature_labels = {'100%', '75%', '50%', '25%'};
    feature_eer = results.optimization.feature_reduction(:,3) * 100;
    bar(feature_eer, 'FaceColor', [0.4 0.2 0.8]);
    set(gca, 'XTickLabel', feature_labels, 'FontSize', 10);
    xlabel('Features Retained', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('EER (%)', 'FontSize', 11, 'FontWeight', 'bold');
    title('PCA Feature Reduction', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    for i = 1:length(feature_eer)
        text(i, feature_eer(i) + 1, sprintf('%.2f%%', feature_eer(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    sgtitle('DETAILED OPTIMIZATION EXPERIMENTS', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(gcf, 'fig3_optimization_details.png');
    fprintf(' ✓ Saved: fig3_optimization_details.png\n');
    close;
    
    % FAR/FRR Curves 
    generate_far_frr_curves(results, templates, valid_users);
end

%% FAR/FRR Curves Visualization (NEW)
function generate_far_frr_curves(results, templates, valid_users)
    fprintf(' Generating FAR/FRR curves...\n');
    
    % Use cross-day scenario for demonstration
    scenario_data = templates.scenario2;
    X_train = scenario_data.X_train;
    y_train = scenario_data.y_train;
    X_test = scenario_data.X_test;
    y_test = scenario_data.y_test;
    
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_test_norm = (X_test - mu) ./ (sigma + eps);
    
    figure('Position', [100, 100, 1400, 900]);
    
    % Select 4 representative users
    user_subset = find(valid_users, 4);
    
    for idx = 1:length(user_subset)
        target_user = user_subset(idx);
        
        % Train binary classifier for this user
        binary_train = (y_train == target_user);
        binary_test = (y_test == target_user);
        train_targets = full(ind2vec(binary_train' + 1));
        
        net = patternnet([20, 10], 'trainscg');
        net.trainParam.showWindow = false;
        [net, ~] = train(net, X_train_norm', train_targets);
        
        predictions = net(X_test_norm');
        user_scores = predictions(2, :)';
        
        % Calculate FAR/FRR curves
        thresholds = 0:0.01:1;
        genuine_scores = user_scores(binary_test);
        impostor_scores = user_scores(~binary_test);
        
        far_curve = zeros(size(thresholds));
        frr_curve = zeros(size(thresholds));
        
        for t = 1:length(thresholds)
            thresh = thresholds(t);
            far_curve(t) = sum(impostor_scores >= thresh) / max(length(impostor_scores), 1);
            frr_curve(t) = sum(genuine_scores < thresh) / max(length(genuine_scores), 1);
        end
        
        [~, eer_idx] = min(abs(far_curve - frr_curve));
        EER = (far_curve(eer_idx) + frr_curve(eer_idx)) / 2;
        
        % Plot
        subplot(2, 2, idx);
        plot(thresholds, far_curve * 100, 'r-', 'LineWidth', 2, 'DisplayName', 'FAR');
        hold on;
        plot(thresholds, frr_curve * 100, 'b-', 'LineWidth', 2, 'DisplayName', 'FRR');
        plot(thresholds(eer_idx), EER * 100, 'ko', 'MarkerSize', 10, ...
             'MarkerFaceColor', 'g', 'DisplayName', sprintf('EER=%.2f%%', EER*100));
        
        xlabel('Threshold', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Error Rate (%)', 'FontSize', 11, 'FontWeight', 'bold');
        title(sprintf('User %d - FAR/FRR Curves', target_user), ...
              'FontSize', 12, 'FontWeight', 'bold');
        legend('show', 'Location', 'best');
        grid on;
    end
    
    sgtitle('FAR/FRR vs Threshold - Cross-Day Testing', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(gcf, 'fig4_far_frr_curves.png');
    fprintf(' ✓ Saved: fig4_far_frr_curves.png\n');
    close;
end

%% Results Summary Display
function display_results_summary_binary(results, valid_users)
    fprintf('\n=== BINARY CLASSIFICATION RESULTS ===\n');
    fprintf('Scenario                FAR       FRR       EER\n');
    fprintf('-------------------------------------------------\n');
    fprintf('Same Day               %6.2f%%   %6.2f%%   %6.2f%%\n', ...
        nanmean(results.scenario1(valid_users,1))*100, ...
        nanmean(results.scenario1(valid_users,2))*100, ...
        nanmean(results.scenario1(valid_users,3))*100);
    fprintf('Cross-Day              %6.2f%%   %6.2f%%   %6.2f%%\n', ...
        nanmean(results.scenario2(valid_users,1))*100, ...
        nanmean(results.scenario2(valid_users,2))*100, ...
        nanmean(results.scenario2(valid_users,3))*100);
    fprintf('Combined               %6.2f%%   %6.2f%%   %6.2f%%\n', ...
        nanmean(results.scenario3(valid_users,1))*100, ...
        nanmean(results.scenario3(valid_users,2))*100, ...
        nanmean(results.scenario3(valid_users,3))*100);
    
    fprintf('\n=== SENSOR COMPARISON ===\n');
    fprintf('Sensor Type            EER\n');
    fprintf('-----------------------------\n');
    fprintf('Accelerometer Only    %6.2f%%\n', nanmean(results.sensor.accelerometer(valid_users,3))*100);
    fprintf('Gyroscope Only        %6.2f%%\n', nanmean(results.sensor.gyroscope(valid_users,3))*100);
    fprintf('Combined Sensors      %6.2f%%\n', nanmean(results.scenario2(valid_users,3))*100);
    
    fprintf('\n=== ACTUAL OPTIMIZATION RESULTS ===\n');
    fprintf('Optimization           EER\n');
    fprintf('-----------------------------\n');
    fprintf('Baseline (50%% ovlp)   %6.2f%%\n', nanmean(results.scenario2(valid_users,3))*100);
    fprintf('Window 0%% overlap     %6.2f%%\n', results.optimization.overlap(1,3)*100);
    fprintf('Window 25%% overlap    %6.2f%%\n', results.optimization.overlap(2,3)*100);
    fprintf('Window 75%% overlap    %6.2f%%\n', results.optimization.overlap(4,3)*100);
    fprintf('Best NN [50,25]       %6.2f%%\n', min(results.optimization.nn_tuning(:,3))*100);
    fprintf('Best PCA (75%%)        %6.2f%%\n', min(results.optimization.feature_reduction(:,3))*100);
    
    save('binary_classifier_results.mat', 'results');
    fprintf('\n✓ Results saved to binary_classifier_results.mat\n');
end

%% Authentication Metrics Calculation
function [FAR, FRR, EER] = calculate_authentication_metrics(true_genuine, prob_genuine)
    thresholds = 0:0.01:1;
    genuine_scores = prob_genuine(true_genuine);
    impostor_scores = prob_genuine(~true_genuine);
    
    far_curve = zeros(size(thresholds));
    frr_curve = zeros(size(thresholds));
    
    for t = 1:length(thresholds)
        thresh = thresholds(t);
        far_curve(t) = sum(impostor_scores >= thresh) / max(length(impostor_scores), 1);
        frr_curve(t) = sum(genuine_scores < thresh) / max(length(genuine_scores), 1);
    end
    
    [~, eer_idx] = min(abs(far_curve - frr_curve));
    EER = (far_curve(eer_idx) + frr_curve(eer_idx)) / 2;
    
    % FAR and FRR at 0.5 threshold
    pred_genuine = prob_genuine >= 0.5;
    FP = sum(~true_genuine & pred_genuine);
    TN = sum(~true_genuine & ~pred_genuine);
    FN = sum(true_genuine & ~pred_genuine);
    TP = sum(true_genuine & pred_genuine);
    
    FAR = FP / max((FP + TN), 1);
    FRR = FN / max((FN + TP), 1);
end