function [FD_features, MD_features, raw_data_info] = feature_extraction(overlap_percent)

if nargin < 1
    overlap_percent = 50; % Default 50% overlap
end

fprintf('\n========================================\n');
fprintf('SCRIPT 1: FEATURE EXTRACTION\n');
fprintf('Window Overlap: %d%%\n', overlap_percent);
fprintf('========================================\n');

rng(42); % Reproducibility

num_users = 10;
FD_user_data = cell(num_users, 1);
MD_user_data = cell(num_users, 1);

%% --- DATA LOADING -------
fprintf('\n[Loading First Day Data - Training Set]\n');
for u = 1:num_users
    filename = sprintf('U%dNW_FD.csv', u);
    if exist(filename, 'file')
        try
            temp_data = readmatrix(filename);
            FD_user_data{u} = temp_data(:, 2:7);
            fprintf(' ✓ User %2d: %6d samples loaded\n', u, size(FD_user_data{u}, 1));
        catch ME
            warning(ME.identifier, '%s', ME.message);
            FD_user_data{u} = [];
        end
    else
        warning('File not found: %s', filename);
        FD_user_data{u} = [];
    end
end

fprintf('\n[Loading More Day Data - Testing Set]\n');
for u = 1:num_users
    filename = sprintf('U%dNW_MD.csv', u);
    if exist(filename, 'file')
        try
            temp_data = readmatrix(filename);
            MD_user_data{u} = temp_data(:, 2:7);
            fprintf(' ✓ User %2d: %6d samples loaded\n', u, size(MD_user_data{u}, 1));
        catch ME
            warning(ME.identifier, '%s', ME.message);
            MD_user_data{u} = [];
        end
    else
        warning('File not found: %s', filename);
        MD_user_data{u} = [];
    end
end

% Store raw data info
raw_data_info = struct();
raw_data_info.FD_data = FD_user_data;
raw_data_info.MD_data = MD_user_data;

%% --- DATA VISUALIZATION (RUNS ONLY ONCE) ----------
persistent visualization_done;
if isempty(visualization_done)
    visualization_done = false;
end

if ~visualization_done && overlap_percent == 50
    fprintf('\n[Generating Data Visualization]\n');
    visualization_done = true;  % prevent re-run

    try
        figure('Position', [100, 100, 1200, 800]);

        user_for_viz = 1;
        if ~isempty(FD_user_data{user_for_viz}) && size(FD_user_data{user_for_viz}, 1) >= 300
            sample_duration = 300;
            sample_data = FD_user_data{user_for_viz}(1:sample_duration, 1:3);
            time_axis = (1:sample_duration) / 30;

            subplot(2, 1, 1);
            plot(time_axis, sample_data(:, 1), 'r-', 'LineWidth', 1.5);
            hold on;
            plot(time_axis, sample_data(:, 2), 'g-', 'LineWidth', 1.5);
            plot(time_axis, sample_data(:, 3), 'b-', 'LineWidth', 1.5);
            xlabel('Time (s)'); ylabel('Acceleration');
            title('Accelerometer Signals (User 1)');
            grid on;

            subplot(2, 1, 2);
            sample_gyro = FD_user_data{user_for_viz}(1:sample_duration, 4:6);
            plot(time_axis, sample_gyro(:, 1), 'r-', 'LineWidth', 1.5); hold on;
            plot(time_axis, sample_gyro(:, 2), 'g-', 'LineWidth', 1.5);
            plot(time_axis, sample_gyro(:, 3), 'b-', 'LineWidth', 1.5);
            xlabel('Time (s)'); ylabel('Gyroscope');
            title('Gyroscope Signals (User 1)');
            grid on;

            sgtitle('Raw Sensor Data Visualization');
            saveas(gcf, 'fig1_raw_sensor_data.png');
            fprintf(' ✓ Saved: fig1_raw_sensor_data.png\n');
            close;
        end
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end
end

%% --- SEGMENTATION ------------
fprintf('\n[Segmentation - %d%% Overlap]\n', overlap_percent);

window_size = 150;
overlap_samples = round(window_size * overlap_percent / 100);
stride = window_size - overlap_samples;

fprintf('Window: %d | Overlap: %d | Stride: %d\n', window_size, overlap_samples, stride);

FD_segments = cell(num_users, 1);
MD_segments = cell(num_users, 1);

fprintf('\n[Processing Training Segments]\n');
for u = 1:num_users
    if ~isempty(FD_user_data{u})
        [seg, ok] = segment_and_interpolate(FD_user_data{u}, window_size, stride);
        FD_segments{u} = seg;
        if ok
            fprintf(' ✓ User %2d: %d segments\n', u, size(seg, 1));
        else
            fprintf(' ✗ User %2d: Segmentation failed\n', u);
        end
    end
end

fprintf('\n[Processing Testing Segments]\n');
for u = 1:num_users
    if ~isempty(MD_user_data{u})
        [seg, ok] = segment_and_interpolate(MD_user_data{u}, window_size, stride);
        MD_segments{u} = seg;
        if ok
            fprintf(' ✓ User %2d: %d segments\n', u, size(seg, 1));
        else
            fprintf(' ✗ User %2d: Segmentation failed\n', u);
        end
    end
end

%% --- FEATURE EXTRACTION -----------------------------------------------
fprintf('\n[Extracting Features]\n');
FD_features = cell(num_users, 1);
MD_features = cell(num_users, 1);

for u = 1:num_users
    if ~isempty(FD_segments{u})
        FD_features{u} = extract_features(FD_segments{u});
        fprintf(' ✓ User %2d FD: %d segments\n', u, size(FD_features{u}, 1));
    end

    if ~isempty(MD_segments{u})
        MD_features{u} = extract_features(MD_segments{u});
        fprintf(' ✓ User %2d MD: %d segments\n', u, size(MD_features{u}, 1));
    end
end

fprintf('\n✓ SCRIPT 1 COMPLETED\n');

end % end main function

%% =======================================================================
%% HELPER FUNCTIONS
%% =======================================================================

function [segments, success] = segment_and_interpolate(data, target_size, stride)
success = true;

try
    num_segments = floor((size(data,1) - target_size) / stride) + 1;

    if num_segments < 1
        warning('Insufficient data for segmentation.');
        segments = [];
        success = false;
        return;
    end

    segments = zeros(num_segments, target_size, size(data,2));

    for i = 1:num_segments
        idx1 = (i-1)*stride + 1;
        idx2 = idx1 + target_size - 1;

        win = data(idx1:idx2, :);

        for c = 1:size(data,2)
            x = 1:size(win,1);
            xq = linspace(1, size(win,1), target_size);
            segments(i,:,c) = interp1(x, win(:,c), xq, 'linear');
        end
    end

catch ME
    warning(ME.identifier, '%s', ME.message);
    segments = [];
    success = false;
end
end

function features = extract_features(segments)
num_segments = size(segments,1);
num_sensors = size(segments,3);
numF = 13;

features = zeros(num_segments, num_sensors*numF);

for s = 1:num_segments
    for j = 1:num_sensors
        sig = squeeze(segments(s,:,j));
        base = (j-1)*numF + 1;
        features(s,base:base+numF-1) = [
            mean(sig), std(sig), min(sig), max(sig), range(sig), ...
            rms(sig), var(sig), median(sig), iqr(sig), ...
            skewness(sig), kurtosis(sig), sum(sig.^2), calculate_entropy(sig)
        ];
    end
end

end

function H = calculate_entropy(signal)
normalized = (signal - min(signal)) / (max(signal)-min(signal) + eps);
[counts, ~] = histcounts(normalized, 10);
p = counts / sum(counts);
p = p(p > 0);
H = -sum(p .* log2(p));
end
