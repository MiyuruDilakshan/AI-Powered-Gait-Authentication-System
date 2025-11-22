function templates = template_generation(FD_features, MD_features)

fprintf('\n========================================\n');
fprintf('SCRIPT 2: TEMPLATE GENERATION (BINARY)\n');
fprintf('========================================\n');

num_users = length(FD_features);
templates = struct();
templates.num_users = num_users;

% Identify valid users (those with data)
valid_users = false(num_users, 1);
for u = 1:num_users
    if ~isempty(FD_features{u}) && ~isempty(MD_features{u})
        valid_users(u) = true;
    end
end
templates.valid_users = valid_users;

fprintf('Valid users with complete data: %d/%d\n', sum(valid_users), num_users);

% Store feature indices for sensor comparison
templates.acc_features = 1:39;   % Accelerometer: 3 axes × 13 features
templates.gyro_features = 40:78; % Gyroscope: 3 axes × 13 features

%% SCENARIO 1: Same Day Testing (FD 70% train, 30% test)
fprintf('\n[Scenario 1: Same Day - 70/30 Split]\n');
scenario1 = create_scenario_binary(FD_features, [], valid_users, 'same_day');
templates.scenario1 = scenario1;
fprintf(' ✓ Training: %d samples\n', size(scenario1.X_train, 1));
fprintf(' ✓ Testing: %d samples\n', size(scenario1.X_test, 1));

%% SCENARIO 2: Cross-Day Testing (FD train, MD test)
fprintf('\n[Scenario 2: Cross-Day - FD→MD]\n');
scenario2 = create_scenario_binary(FD_features, MD_features, valid_users, 'cross_day');
templates.scenario2 = scenario2;
fprintf(' ✓ Training: %d samples (First Day)\n', size(scenario2.X_train, 1));
fprintf(' ✓ Testing: %d samples (More Day)\n', size(scenario2.X_test, 1));

%% SCENARIO 3: Combined Random Split (FD+MD 70% train, 30% test)
fprintf('\n[Scenario 3: Combined Random - 70/30 Split]\n');
scenario3 = create_scenario_binary(FD_features, MD_features, valid_users, 'combined');
templates.scenario3 = scenario3;
fprintf(' ✓ Training: %d samples (70%% of FD+MD)\n', size(scenario3.X_train, 1));
fprintf(' ✓ Testing: %d samples (30%% of FD+MD)\n', size(scenario3.X_test, 1));

fprintf('\n✓ SCRIPT 2 COMPLETED: Binary classification templates generated\n');

end

%% Create Binary Classification Scenarios
function scenario = create_scenario_binary(FD_features, MD_features, valid_users, scenario_type)
    num_users = length(FD_features);
    
    if strcmp(scenario_type, 'same_day')
        % Scenario 1: Same day 70/30 split
        all_X = [];
        all_y = [];
        
        for u = 1:num_users
            if valid_users(u) && ~isempty(FD_features{u})
                user_features = FD_features{u};
                user_labels = u * ones(size(user_features, 1), 1);
                
                all_X = [all_X; user_features];
                all_y = [all_y; user_labels];
            end
        end
        
        % Random 70/30 split with stratification
        cv = cvpartition(all_y, 'HoldOut', 0.3);
        X_train = all_X(cv.training, :);
        y_train = all_y(cv.training);
        X_test = all_X(cv.test, :);
        y_test = all_y(cv.test);
        
    elseif strcmp(scenario_type, 'cross_day')
        % Scenario 2: Train on FD, test on MD
        X_train = [];
        y_train = [];
        X_test = [];
        y_test = [];
        
        for u = 1:num_users
            if valid_users(u)
                % All FD for training
                if ~isempty(FD_features{u})
                    X_train = [X_train; FD_features{u}];
                    y_train = [y_train; u * ones(size(FD_features{u}, 1), 1)];
                end
                
                % All MD for testing
                if ~isempty(MD_features{u})
                    X_test = [X_test; MD_features{u}];
                    y_test = [y_test; u * ones(size(MD_features{u}, 1), 1)];
                end
            end
        end
        
    elseif strcmp(scenario_type, 'combined')
        % Scenario 3: Combine FD+MD, then 70/30 split
        all_X = [];
        all_y = [];
        
        for u = 1:num_users
            if valid_users(u)
                if ~isempty(FD_features{u})
                    all_X = [all_X; FD_features{u}];
                    all_y = [all_y; u * ones(size(FD_features{u}, 1), 1)];
                end
                if ~isempty(MD_features{u})
                    all_X = [all_X; MD_features{u}];
                    all_y = [all_y; u * ones(size(MD_features{u}, 1), 1)];
                end
            end
        end
        
        % Random 70/30 split with stratification
        cv = cvpartition(all_y, 'HoldOut', 0.3);
        X_train = all_X(cv.training, :);
        y_train = all_y(cv.training);
        X_test = all_X(cv.test, :);
        y_test = all_y(cv.test);
    end
    
    % Store in structure
    scenario.X_train = X_train;
    scenario.y_train = y_train;
    scenario.X_test = X_test;
    scenario.y_test = y_test;
end