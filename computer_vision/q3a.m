% Define parameters
num_points = 100;
x = 2;  % Original x-coordinate of the vertical line
y = linspace(-5, 5, num_points);  % Range of y-values
kplus = 0.01;  % Distortion parameter - pincushion distortion
kminus = -0.01; % Distortion parameter - barrel distortion

% Compute original radii
r = sqrt(x^2 + y.^2);

% Compute distorted radii
r_d_plus = r ./ sqrt(1 - 2 * kplus * r.^2);
r_d_minus = r ./ sqrt(1 - 2 * kminus * r.^2);

% Compute magnification ratio
mag_ratio_plus = r_d_plus ./ r;
mag_ratio_minus = r_d_minus ./ r;

% Compute distorted x and y coordinates
new_x_plus = x * mag_ratio_plus;
new_y_plus = y .* mag_ratio_plus;

new_x_minus = x * mag_ratio_minus;
new_y_minus = y .* mag_ratio_minus;

% Plot original and distorted lines
figure;
hold on;
plot(x * ones(1, num_points), y, 'b-', 'LineWidth', 2, 'DisplayName', 'Original Line'); % Original vertical line
plot(new_x_plus, new_y_plus, 'r-', 'LineWidth', 2, 'DisplayName', 'Distorted Line, k > 0');
plot(new_x_minus, new_y_minus, 'g-', 'LineWidth', 2, 'DisplayName', 'Distorted Line, k < 0');
plot(0, 0, 'x', 'DisplayName', 'Image Origin')
legend;
legend('Location','northwest');
xlabel('x');
ylabel('y');
xlim([-3, 3])
ylim([-6,6])
title('Effect of Radial Distortion on a Vertical Line');
grid on;
hold off;