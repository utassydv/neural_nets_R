library(keras)
library(here)
library(grid)
library(magick)
library(filesstrings)
library(kableExtra)

####
##Generating validation data from the training data (source: Kata Süle, Cosmin Ticu)
#
#validation_indeces_yes <- sample(list.files(paste0(here(), "/data/hot-dog-not-hot-dog/train/hot_dog")), size = 150, replace = F)
#validation_indeces_no <- sample(list.files(paste0(here(), "/data/hot-dog-not-hot-dog/train/not_hot_dog")), size = 150, replace = F)
#
#dir.create(paste0(here(),"/data/hot-dog-not-hot-dog/validation"))
#dir.create(paste0(here(),"/data/hot-dog-not-hot-dog/validation/hot_dog"))
#dir.create(paste0(here(),"/data/hot-dog-not-hot-dog/validation/not_hot_dog"))
#
## move yes cases
#for (file in list.files(paste0(here(), "/data/hot-dog-not-hot-dog/train/hot_dog"))) {
# if (file %in% validation_indeces_yes){
#   file.move(paste0(here(), "/data/hot-dog-not-hot-dog/train/hot_dog/", file), paste0(here(),"/data/hot-dog-not-hot-dog/validation/hot_dog/"))
# }
#}
#
## move no cases
#for (file in list.files(paste0(here(), "/data/hot-dog-not-hot-dog/train/not_hot_dog"))) {
# if (file %in% validation_indeces_no){
#   file.move(paste0(here(), "/data/hot-dog-not-hot-dog/train/not_hot_dog/", file), paste0(here(),"/data/hot-dog-not-hot-dog/validation/not_hot_dog/"))
# }
#}

# Setting parameters for image data generators

train_datagen <-  image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)  
test_datagen <-image_data_generator(rescale = 1/255)  


#Generating data

image_size <- c(150, 150)
batch_size <- 50

train_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/train/"),
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

validation_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/validation/"),   
  validation_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

test_generator <- flow_images_from_directory(
  file.path(here(), "data/hot-dog-not-hot-dog/test/"),
  test_datagen,
  target_size = image_size,
  batch_size = batch_size,
  class_mode = "binary"
)

#Define model1
hot_dog_model1 <- keras_model_sequential() 
hot_dog_model1 %>% 
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

hot_dog_model1 %>% compile(
  loss = "binary_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)

#hotdog_model1_fit <- hot_dog_model1 %>% fit(
#  train_generator,
#  steps_per_epoch = train_generator$n / batch_size,
#  epochs = 20,
#  validation_data = validation_generator,
#  validation_steps = validation_generator$n / batch_size, 
#  callbacks = list(
#    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1))
#)
#saveRDS(hotdog_model1_fit, "models/hotdog_model1_fit.rds")
#save_model_hdf5(hot_dog_model1, "models/hot_dog_model1.h5")
#evaluate(hot_dog_model1, test_generator)

hot_dog_model1 <- load_model_hdf5("models/hot_dog_model1.h5")

#Define model2
#https://github.com/matiascaputti/not-hotdog/blob/master/1-%20Not%20Hotdog%20-%20base.ipynb

hot_dog_model2 <- keras_model_sequential() 
hot_dog_model2 %>% 
  layer_conv_2d(filters = 64,
                kernel_size = c(2, 2), 
                activation = 'relu',
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  layer_conv_2d(filters = 32,
                kernel_size = c(2, 2), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.2) %>%
  
  layer_global_average_pooling_2d() %>% # this converts our 3D feature maps to 1D feature vectors
  layer_dense(units = 1024, activation = 'relu') %>% 
  layer_dropout(rate = 0.4) %>%

  layer_dense(units = 1, activation = "sigmoid")   # for binary

hot_dog_model2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'rmsprop',
  metrics = c("accuracy")
)

#hotdog_model2_fit <- hot_dog_model2 %>% fit(
#  train_generator,
#  steps_per_epoch = train_generator$n / batch_size,
#  epochs = 40,
#  validation_data = validation_generator,
#  validation_steps = validation_generator$n / batch_size,
#  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)
#)
#saveRDS(hotdog_model2_fit, "models/hotdog_model2_fit.rds")
#save_model_hdf5(hot_dog_model2, "models/hot_dog_model2.h5")
#evaluate(hot_dog_model2, test_generator)

hot_dog_model2 <- load_model_hdf5("models/hot_dog_model2.h5")


#DATA AUG1
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
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

#hotdog_model2_fit_aug1 <- hot_dog_model2 %>% fit(
#  train_generator,
#  steps_per_epoch = train_generator$n / batch_size,
#  epochs = 20,
#  validation_data = validation_generator,
#  validation_steps = validation_generator$n / batch_size,
#  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)
#)
#saveRDS(hotdog_model2_fit_aug1, "models/hotdog_model2_fit_aug1.rds")
#save_model_hdf5(hot_dog_model2, "models/hot_dog_model2_aug1.h5")
#evaluate(hot_dog_model2, test_generator)

hot_dog_model2_aug1 <- load_model_hdf5("models/hot_dog_model2_aug1.h5")

#DATA AUG2
train_datagen =  image_data_generator(
  rescale = 1/255,
  rotation_range = 180,
  width_shift_range = 0.4,
  height_shift_range = 0.4,
  shear_range = 0.4,
  zoom_range = 0.4,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

#hotdog_model2_fit_aug1 <- hot_dog_model2 %>% fit(
#  train_generator,
#  steps_per_epoch = train_generator$n / batch_size,
#  epochs = 20,
#  validation_data = validation_generator,
#  validation_steps = validation_generator$n / batch_size,
#  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)
#)
#saveRDS(hotdog_model2_fit_aug2, "models/hotdog_model2_fit_aug2.rds")
#save_model_hdf5(hot_dog_model2, "models/hot_dog_model2_aug2.h5")
#evaluate(hot_dog_model2, test_generator)

hot_dog_model2_aug2 <- load_model_hdf5("models/hot_dog_model2_aug2.h5")

#DATA AUG3
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 90,
  width_shift_range = 0.3,
  height_shift_range = 0.3,
  shear_range = 0.3,
  zoom_range = 0.3,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
train_generator <- flow_images_from_directory(
  file.path(here(), "/data/hot-dog-not-hot-dog/train"), # Target directory  
  train_datagen,              # Data generator
  target_size = image_size,  # Resizes all images to 150 × 150
  batch_size = batch_size,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

#hotdog_model2_fit_aug3 <- hot_dog_model2 %>% fit(
#  train_generator,
#  steps_per_epoch = train_generator$n / batch_size,
#  epochs = 20,
#  validation_data = validation_generator,
#  validation_steps = validation_generator$n / batch_size,
#  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)
#)
#saveRDS(hotdog_model2_fit_aug3, "models/hotdog_model2_fit_aug3.rds")
#save_model_hdf5(hot_dog_model2, "models/hot_dog_model2_aug3.h5")
#evaluate(hot_dog_model2, test_generator)

hot_dog_model2_aug3 <- load_model_hdf5("models/hot_dog_model2_aug3.h5")

model_eval_list <- list()
model_eval_list['final_model_no_aug'] <- as.data.frame(evaluate(hot_dog_model2, validation_generator))
model_eval_list['final_model_aug1'] <- as.data.frame(evaluate(hot_dog_model2_aug1, validation_generator))
model_eval_list['final_model_aug2'] <- as.data.frame(evaluate(hot_dog_model2_aug2, validation_generator))
model_eval_list['final_model_aug3'] <- as.data.frame(evaluate(hot_dog_model2_aug3, validation_generator))

model_eval_list_test <- list()
model_eval_list_test['final_model_no_aug'] <- as.data.frame(evaluate(hot_dog_model2, test_generator))
model_eval_list_test['final_model_aug1'] <- as.data.frame(evaluate(hot_dog_model2_aug1, test_generator))
model_eval_list_test['final_model_aug2'] <- as.data.frame(evaluate(hot_dog_model2_aug2, test_generator))
model_eval_list_test['final_model_aug3'] <- as.data.frame(evaluate(hot_dog_model2_aug3, test_generator))


# PRE TRAINED MODELS
 
#N 1
# imagenetet w INCEPLTION_V3
conv_base <- application_inception_v3(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(image_size, 3)
)

#summary(conv_base)
freeze_weights(conv_base)

hot_dog_model2_inception_v3 <- keras_model_sequential()%>%
  conv_base %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

hot_dog_model2_inception_v3 %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'rmsprop',
  metrics = c("accuracy")
)

#hot_dog_model2_inception_v3_fit <- hot_dog_model2_inception_v3 %>% fit(
#  train_generator,
#  steps_per_epoch = train_generator$n / batch_size, 
#  epochs = 15,
#  validation_data = validation_generator,
#  validation_steps = validation_generator$n / batch_size, 
#  callbacks = list(
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 3))
#)
#saveRDS(hot_dog_model2_inception_v3_fit, "models/hot_dog_model2_inception_v3_fit.rds")
#save_model_hdf5(hot_dog_model2_inception_v3, "models/hot_dog_model2_inception_v3.h5")
#evaluate(hot_dog_model2_inception_v3, test_generator)

hot_dog_model2_inception_v3 <- load_model_hdf5("models/hot_dog_model2_inception_v3.h5")

model_eval_list['final_imagenet_inception_v3'] <- as.data.frame(evaluate(hot_dog_model2_inception_v3, validation_generator))
model_eval_list_test['final_imagenet_inception_v3'] <- as.data.frame(evaluate(hot_dog_model2_inception_v3, test_generator))


#N 2
# imagenetet w MOBILENET
base_model <- application_mobilenet(
  weights = 'imagenet', 
  include_top = FALSE,
  input_shape = c(image_size, 3))

#summary(base_model)
freeze_weights(base_model)

hot_dog_model2_mobilenet <- keras_model_sequential()%>%
  base_model %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid')

hot_dog_model2_mobilenet %>% compile(
  loss = "binary_crossentropy",
  optimizer = 'rmsprop',
  metrics = c("accuracy")
)

#hot_dog_model2_mobilenet_fit <- hot_dog_model2_mobilenet %>% fit(
# train_generator,
# steps_per_epoch = train_generator$n / batch_size,
# epochs = 10,
# validation_data = validation_generator,
# validation_steps = validation_generator$n / batch_size, 
# callbacks = list(
#   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 5))
#)
#saveRDS(hot_dog_model2_mobilenet_fit, "models/hot_dog_model2_mobilenet_fit.rds")
#save_model_hdf5(hot_dog_model2_mobilenet, "models/hot_dog_model2_mobilenet.h5")
#evaluate(hot_dog_model2_mobilenet, test_generator)

hot_dog_model2_mobilenet <- load_model_hdf5("models/hot_dog_model2_mobilenet.h5")

model_eval_list['final_imagenet_mobilenet'] <- as.data.frame(evaluate(hot_dog_model2_mobilenet, validation_generator))
model_eval_list_test['final_imagenet_mobilenet_test'] <- as.data.frame(evaluate(hot_dog_model2_mobilenet, test_generator))

res_valid <- t(as.data.frame(model_eval_list))
res_test <- t(as.data.frame(model_eval_list_test))


colnames(res_valid) <- c('loss','accuracy')
colnames(res_test) <- c('loss','accuracy')

knitr::kable(res_valid, caption = "", digits = 3 ) %>% 
  kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))



