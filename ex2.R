library(keras)
library(here)
library(grid)
library(magick)


example_image_path <- file.path(here(), "/data/hot-dog-not-hot-dog/train/hot_dog/1000288.jpg")
image_read(example_image_path)  # this is a PIL image

img <- image_load(example_image_path, target_size = c(150, 150))  # this is a PIL image
x <- image_to_array(img) / 255
grid::grid.raster(x)


train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
# Note that the validation data shouldn't be augmented!
validation_datagen <- image_data_generator(rescale = 1/255)  
test_datagen <- image_data_generator(rescale = 1/255) 

xx <- flow_images_from_data(
  array_reshape(x * 255, c(1, dim(x))),  # take the previous image as base, multiplication is only to conform with the image generator's rescale parameter
  generator = train_datagen
)
augmented_versions <- lapply(1:10, function(ix) generator_next(xx) %>%  {.[1, , , ]})
# see examples by running in console:
grid::grid.raster(augmented_versions[[1]])


image_size <- c(150, 150)
batch_size <- 50

train_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/train/"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

validation_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/test/"),   
  validation_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

### NO VALID
test_generator <- flow_images_from_directory(
  file.path(here(), "data/dogs-vs-cats/test/"), # Target directory  
  test_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

hot_dog_model <- keras_model_sequential() 
hot_dog_model %>% 
  layer_conv_2d(filters = 32,
                kernel_size = c(3, 3), 
                activation = 'relu',
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16,
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 8, activation = 'relu') %>% 
  layer_dense(units = 1, activation = "sigmoid")   # for binary

hot_dog_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

hot_dog_model_hist <- hot_dog_model %>% fit(
  train_generator,
  steps_per_epoch = 500 / batch_size,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

saveRDS(hot_dog_model_hist, "models/ex2_cnn_fit.rds")
save_model_hdf5(hot_dog_model, "models/ex2_cnn.h5")
