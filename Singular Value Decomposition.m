% Task 1: Load an image of your choice into MATLAB. Turn it into grayscale. Plot the initial grayscale image.
% Load an image of the minions
image1 = imread("minions2.jpg");
figure;
imshow(image1)

% Convert image to grayscale
image2 = rgb2gray(image1);
figure;
imshow(image2)
title('Original Image')


% Task 2: Add three types of noise to the image [adding noise to an image]. The result will be three images, each with their own noise type. Plot the noisy images.
% Add 3 types of noise to the image
% Noise 1: Gaussian noise
figure;
image_noise_gauss = imnoise(image2, 'gaussian');
subplot(1,3,1)
imshow(image_noise_gauss)
title('Image with Gaussian noise.')

% Noise 2: Poisson noise
image_noise_poisson = imnoise(image2, 'poisson');
subplot(1,3,2)
imshow(image_noise_poisson)
title('Image with Poisson noise.')

% Noise 3: Salt & Pepper noise
image_noise_saltpepper = imnoise(image2, 'salt & pepper');
subplot(1,3,3)
imshow(image_noise_saltpepper)
title('Image with Salt & Pepper noise.')


% Task 3: De-construct the noisy images with Singular Value Decomposition and plot the singular values as cumulative sum. Is there any difference between the singular values of the images with different noise?
% Deconstructing the noisy images using svd
% Gaussian noise 
[U_gauss, S_gauss, V_gauss] = svd(double(image_noise_gauss));
s_gauss = diag(S_gauss);
figure;
subplot(1,3,1)
plot(1:length(s_gauss), cumsum(s_gauss))
title('Gaussian noise cumsum graph')
xlabel('Singular Values')
ylabel('Cumulative Sum')

% Poisson noise
[U_poisson, S_poisson, V_poisson] = svd(double(image_noise_poisson));
s_poisson = diag(S_poisson);
subplot(1,3,2)
plot(1:length(s_poisson), cumsum(s_poisson))
title('Poisson noise cumsum graph')
xlabel('Singular Values')
ylabel('Cumulative Sum')

% Salt & Pepper noise
[U_saltpepper, S_saltpepper, V_saltpepper] = svd(double(image_noise_saltpepper));
s_saltpepper = diag(S_saltpepper);
subplot(1,3,3)
plot(1:length(s_saltpepper), cumsum(s_saltpepper))
title('Salt & Pepper noise cumsum graph')
xlabel('Singular Values')
ylabel('Cumulative Sum')


% Answer question 3: Is there any difference between the singular values of the images with different noise?
disp(['The singular values of the images with different noise have relatively similar magnitude, but they have different patterns of decay. ' ...
    'The Gaussian noise image decays in a smooth and steady way, while the Poisson noise image decays relatively more quickly and suddenly. ' ...
    'The Salt & Pepper noise image has some spikes in the singular values that correspond to the high contrast pixels in the image.']);


% Create a vector of singular values from 1 to 409 with a step size of 5
SingularVals = 1:5:409;

% Initialize three vectors to store the RMSE values for each noise type
r_gauss = zeros(1,length(SingularVals));
r_poisson = zeros(1,length(SingularVals));
r_saltpepper = zeros(1,length(SingularVals));

figure;
count = 0;
for k = SingularVals
    count = count + 1;
    % Reconstructing the noisy images using k singular values
    image_noise_gauss_r = U_gauss(:, 1:k) * S_gauss(1:k, 1:k) * V_gauss(:, 1:k)';
    % Calculate the RMSE using the immse function
    r_gauss(count) = immse(double(image_noise_gauss_r), double(image2));

    % Poisson noise reconstruct
    image_noise_poisson_r = U_poisson(:, 1:k) * S_poisson(1:k, 1:k) * V_poisson(:, 1:k)';
    % Calculate the RMSE using the immse function
    r_poisson(count) = immse(double(image_noise_poisson_r), double(image2));

    % Salt & Pepper noise reconstruct
    image_noise_saltpepper_r = U_saltpepper(:, 1:k) * S_saltpepper(1:k, 1:k) * V_saltpepper(:, 1:k)';
    % Calculate the RMSE using the immse function
    r_saltpepper(count) = immse(double(image_noise_saltpepper_r), double(image2));
end

% Plot of no. of singular values against RMSE
figure;
plot(SingularVals, r_gauss, SingularVals, r_poisson, SingularVals, r_saltpepper)
title('RMSE vs Number of Singular Values')
xlabel('Number of Singular Values')
ylabel('RMSE')
legend('Gaussian Noise', 'Poisson Noise', 'Salt & Pepper Noise')

% Task 5: Can you obtain noise reduction by reconstructing the image with an optimal number of singular values? What is the optimal value in each case?
% Answer question 5: Can you obtain noise reduction by reconstructing the image with an optimal number of singular values? What is the optimal value in each case?
disp(['Yes, we can obtain noise reduction by reconstructing the image with an optimal number of singular values. This is because we can ' ...
    'preserve most of the information in the original image while discarding some of the noise components that have smaller singular ' ...
    'values. The optimal number of singular values for each noise type is different, and it can be found by looking at the RMSE plot ' ...
    'and finding the minimum point for each curve.']);
% Find the optimal number of singular values for each noise type by finding the minimum RMSE value for each curve
[~, idx_gauss] = min(r_gauss);
[~, idx_poisson] = min(r_poisson);
[~, idx_saltpepper] = min(r_saltpepper);
% Display the optimal number of singular values for each noise type
disp(['For Gaussian noise, the optimal number of singular values is ', num2str(SingularVals(idx_gauss)), '.']);
disp(['For Poisson noise, the optimal number of singular values is ', num2str(SingularVals(idx_poisson)), '.']);
disp(['For Salt & Pepper noise, the optimal number of singular values is ', num2str(SingularVals(idx_saltpepper)), '.']);

% Task 6: How much have you compressed the image by selecting the appropriate number of singular values?
% Answer question 6: How much have you compressed the image by selecting the appropriate number of singular values?
disp(['To find the compression ratio, we can compare how much space the original image takes and how much space the reconstructed image takes when we use only the optimal number of singular values. The compression ratio is (mn)/(k(m+n+1)), where m and n are the height and width of the original image, and k is the optimal number of singular values.']);
% Get the dimensions of the original image
[m, n] = size(image2);
% Calculate and display the compression ratio for each noise type
compression_ratio_gauss = (m*n)/(SingularVals(idx_gauss)*(m+n+1));
compression_ratio_poisson = (m*n)/(SingularVals(idx_poisson)*(m+n+1));
compression_ratio_saltpepper = (m*n)/(SingularVals(idx_saltpepper)*(m+n+1));
disp(['The compression ratio for Gaussian noise is ', num2str(compression_ratio_gauss), ', which means we have compressed the image by about ', num2str(100 - compression_ratio_gauss*100), '%.']);
disp(['The compression ratio for Poisson noise is ', num2str(compression_ratio_poisson), ', which means we have compressed the image by about ', num2str(100 - compression_ratio_poisson*100), '%.']);
disp(['The compression ratio for Salt & Pepper noise is ', num2str(compression_ratio_saltpepper), ', which means we have compressed the image by about ', num2str(100 - compression_ratio_saltpepper*100), '%.'])