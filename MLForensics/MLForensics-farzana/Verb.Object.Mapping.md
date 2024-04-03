# Map Code Snippets to Verb Object Pairs 

## Akond Rahman 

### Needed to write methodology 

Use the following table to find what are verb object pairs 

### Mapping 
| Code Snippet | Verb | Object | Event | Final Category |
|--------------|------|--------------------|---------------------|-------------------|
| torch.load() | load | data.load_model_dir|   Load training data from a directory| Load training data|       
| data.load()  | load | data.dset_dir      |   Load training data from a directory| Load training data|       
| pickle.load()| load | fp                 |   Load training data from a file     | Load training data|       
| json.load()  | load | config_file        |   Load training data from a file     | Load training data|       
| np.load()    | load | path               |   Load training data| Load training data|       
| pickle.load()| load | fp                 |   Load training data| Load training data|       
| blob.upload_from_filename()| upload_from_filename | self.local_save_file | Upload training data from local file| Load training data|   
| yaml.load()  | load | f                  |   Load training data| Load training data|       
| hub.load()   | load | params.hub_module_url|   Load training data from remote source | Load training data|       
| data_loader_factory.get_data_loader().load()| load | dataloader_params, input_context |   Load training data| Load training data|       
| tf.io.read_file()   | read_file | filename|   Load training data from local file | Load training data| 
| tf.data.Dataset.from_tensor_slices() | from_tensor_slices() | filenames|   Load training data from local file | Load training data| 
| TaggingDataLoader().load()   | load | data_config|   Load training data from local file | Load training data| 
| pd.read_csv()   | read_csv | attributes_file|   Load training data from local CSV file | Load training data| 
| ibrosa.load()   | load() | filename|   Load training data from local CSV file | Load training data| 
| dset.MNIST()    | MNIST() | .|   Load training data from local directory | Load training data| 
| tarfile.open()  | open() | target_file|   Load training data from local directory | Load training data| 
| audio.load_wav()| load_wav() | wav_path|   Load training data from local directory | Load training data| 
| Image.open() | open() | args.demo_image|   Load training data from local directory | Load training data| 
| agent.replay_buffer.load()  | load() | self.rbuf_filename|   Load training data from local directory | Load training data| 
| h5py.File()  | File() | hdf5_file|   Load training data of H5 binary type from local directory | Load training data| 
| np.frombuffer()  | frombuffer() | f.read|   Load training data from local directory | Load training data| 
| get_loader()  | get_loader() | config |   Load training data from local directory | Load training data| 
| load_randomly_augmented_audio()  | load_randomly_augmented_audio() | audio_path |   Load  audio data for training from local directory | Load training data| 
| open()  | open() | input_file |   Load  file for training from local directory | Load training data| 
| open()  | open() | args.wavenet_params |   Load wavenet file for training from local directory | Load training data| 
| load_generic_audio(self.audio_dir)  | load_generic_audio() | self.audio_dir |   Load audio file for training  | Load training data|
| load_audio(args.input_path)  | load_audio() | args.input_path |   Load audio file for training  | Load training data| 
| load_audio()  | load_audio() | dset |   Load image file for training  | Load training data| 
| _load_vocab_file()  | _load_vocab_file() | vocab_file |   Load vocabulary file file for training  | Load training data| 
| read_h5file(os.path.join()   | read_h5file() | train.h5 |  Load H5 binary file file for training  | Load training data| 
| DeepSpeech.load_model_package()   | load_model_package() | package |  Load pre-trained model from file | Load pre-trained model | 
| tf.keras.models.load_model()   | load_model() | model_weights_path |  Load pre-trained Terraform model from file | Load pre-trained model | 
| model.load_state_dict(torch.load())    | load_state_dict() | data.load_model_dir |  Load pre-trained model state from directory | Load pre-trained model |
| caffe_parser.read_caffemodel()    | read_caffemodel() | XXX |  Load pre-trained Caffe model from file | Load pre-trained model | 
| vgg.load_from_npy_file()     | load_from_npy_file() | XXX |  Load pre-trained VGG Neural Network model from NPY file | Load pre-trained model |
| load_model(cls, path)     | load_model() | cls, path |  Load pre-trained model from file | Load pre-trained model | 
| load_decoder(labels, cfg= LMConfig)     | load_decoder() | labels |  Load pre-trained DNN decoder  | Load pre-trained model |
| load_previous_values(results_state)    | load_previous_values() | results_state |  Load pre-trained model state | Load pre-trained model | 
| load_pretrained(model, num_classes, settings)  | load_pretrained() | model |  Load pre-trained model state | Load pre-trained model | 
| load_param(prefix, begin_epoch, convert=True) | load_param() | prefix |  Load pre-trained model parameters | Load pre-trained model | 
| model = SeqLabel(data), args1, auxs1 = load_checkpoint(prefix1, epoch1)  | load_checkpoint() | prefix1 |  Load model checkpoint | Load pre-trained model | 
| self.sp_model.Load(sp_model_file)  | sp_model.Load() | sp_model_file |  Load model  | Load pre-trained model | 
| wget.download('http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz')  | wget.download() | 'http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz' |  Download TAR file from remote source  | Download data from remote source | 
| urllib.request.urlopen()  | urlopen() | request |  Download file from remote source using URL  | Download data from remote source | 
| model_zoo.load_url(url)  | load_url() | url |  Download file from remote source using URL  | Download data from remote source | 
| agent.load(misc.download_model())  | misc.download_model() | XXX |  Download file from remote source using URL  | Download data from remote source | 
| latest_blob.download_to_filename(self.local_save_file)  | latest_blob.download_to_filename() | self.local_save_file |  Download file from remote source using URL  | Download data from remote source | 
| _download(filename, working_directory, url_source)  | _download() | filename, working_directory, url_source |  Download file from remote source using URL  | Download data from remote source | 
| download_from_url(path, url)  | download_from_url() | path, url |  Download file from remote source using URL  | Download data from remote source | 
| train_data, train_label = read_h5file(os.path.join(os.getcwd(), 'train.h5'))  | read_h5file() | 'train.h5' |  Load classification labels from H5 binary file  | Load classification labels from file |
| val_data, val_label = read_h5file(os.path.join(os.getcwd(), 'val.h5'))  | read_h5file() | 'val.h5' |  Load classification labels from H5 binary file  | Load classification labels from file |
| label = np.array(hf.get('label'))   | hf.get() | 'label' |  Load classification labels from local file  | Load classification labels from file |
| label = load_image(f).convert('P')    | load_image(f) | 'f' |  Load classification labels from image file  | Load classification labels from file |
| label = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(self.nyud_dir, idx))['segmentation'].astype(np.uint8)    | scipy.io.loadmat('{}/segmentaion/') | '{}/segmentaion/' |  Load classification labels as a matrix  | Load classification labels from file |
|  raw_data,raw_label = load_data_and_labels(fenci_right_save_path,fenci_wrong_save_path)   | load_data_and_labels(fenci_right_save_path,fenci_wrong_save_path) | fenci_right_save_path,fenci_wrong_save_path |  Load classification labels and training data  | Load classification labels from file |
|  label = hfw.create_dataset("labels", data=df_attr[list_col_labels].values)  | hfw.create_dataset() | 'labels' |  Load classification labels from dataset  | Load classification labels from file |
|  output_data = interpreter.get_tensor(output_details[0]['index'])  | interpreter.get_tensor() | output_details[0]['index'] |  Load classification output from dataset  | Load classification labels from file |
|  pred_scores = evaluate(data, model, name, data.nbest)  | evaluate() | data, model, name, data.nbest |  Load classification output from dataset  | Load classification labels from file |
|  coco_gt.loadRes(predictions=coco_predictions)  | loadRes()  | predictions=coco_predictions |  Load classification output from dataset  | Load classification labels from file |
|  argparse.ArgumentParser(description='Input pipeline') | ArgumentParser()  | description='Input pipeline' |  Load pipeline configuration from a file | Load pipeline configuration |
|  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()  | TrainEvalPipelineConfig()  | XXX |  Load pipeline configuration from a file | Load pipeline configuration |
|  get_configs_from_pipeline_file(FLAGS.pipeline_config_path) | get_configs_from_pipeline_file()  | FLAGS.pipeline_config_path |  Load pipeline configuration from a file | Load piepeline configuration |
| obs, reward, done, tt =  env.step(action) | env.step()  | action |  Update in reinforcement learning environment to get reward | Update in reinforcement learning environment |
|  env.step(action) | env.step()  | action |  Update in reinforcement learning environment to get reward | Update in reinforcement learning environment |
|  env = gym.make(params.env_name)  | gym.make() | params.env_name |  Create reinforcement learning environment | Create reinforcement learning environment |
|  policy = torch.load(args.model_path)   | torch.load()  | args.model_path |  Read policy data to create reinforcement learning environment | Create reinforcement learning environment |
|  o, r, done, info = self.env.step(action)  | self.env.step()  | action |  Update in reinforcement learning environment to get info | Update in reinforcement learning environment |
|  obs, reward, done, info = env.step(action)  | env.step()  | action |  Update in reinforcement learning environment to get info | Update in reinforcement learning environment |
|  next_state, reward, done, _ = env.step(action)   | env.step()  | action |  Update in reinforcement learning environment to get info | Update in reinforcement learning environment |
|  batch_ob, reward, done, info = self.env.step(action)   | self.env.step()  | action |  Update in reinforcement learning environment to get info | Update in reinforcement learning environment |
|  state, reward, done, _ = env.step(action)  | env.step()  | action |  Update in reinforcement learning environment to get info | Update in reinforcement learning environment |
|  inner_next_obs, inner_reward, done = wrapped_env.step(action)  | wrapped_env.step()  | action |  Update in reinforcement learning environment to get info | Update in reinforcement learning environment |
|  model.compile()   | model.compile()  | XXX |  Create a NN model without paramters | Create a model based on neural network |
|  model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])   | model.compile()  | ptimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef] |  Create a NN model with paramters | Create a model based on neural network |
|  res = graph.predict({'input':X_check}, verbose=2)   | graph.predict()  | {'input':X_check}, verbose=2 |  Create a NN graph model with paramters | Create a model based on neural network |
|  history = graph.fit()   | graph.fit()  | {'input':X_train, 'output1':y_train, 'output2':y2_train}, nb_epoch=1000, verbose=2 |  Create a NN graph model with paramters | Create a model based on neural network |
|  logs = model.fit()   | model.fit()  | dataset, epochs=1, steps_per_epoch=2 |  Create a NN graph model with paramters | Create a model based on neural network |
|  score = model_from_file.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)   | model_from_file.evaluate()  | X_test, Y_test, show_accuracy=True, verbose=0 |  Create a NN graph model with paramters | Create a model based on neural network |
|  G=model(D, label_images)    | model()  | D, label_images|  Create a NN graph model with paramters | Create a model based on neural network |
|  model = Model(inputs=[inputs], outputs=[conv10])     | Model()  | inputs=[inputs], outputs=[conv10]|  Create a NN graph model with paramters | Create a model based on neural network |
|  data.show_data_summary()   | show_data_summary()  | data|  Show model data summary | Read model results  |
|  prepare_url_image(url)   | prepare_url_image()  | url|  Download external image from URL | Dowload data from external source   |
|  data.show_data_summary()   | show_data_summary()  | data|  Show model data summary | Read model results  |
|  load(saver, sess, restore_from)   | load()  | saver, sess, restore_from|  Load data from external file | Load training data  |
|  data_utils.load_celebA()   | load_celebA()  | img_dim, image_data_format|  Load image data from external file | Load training data  |
|  get_raw_files()   | get_raw_files()  | FLAGS.data_dir, _TEST_DATA_SOURCES|  Load data from external directory | Load training data  |
|  load_attribute_dataset()    | load_attribute_dataset()  | args.attr_file|  Load feature data from external directory | Load training data  |
|  load_lua()     | load_lua()  | args.input_t7|  Load Lua data from external directory | Load training data  |
|  model_dir_path = patch_path('models')     | patch_path()    | 'models'|  Load model data from external directory | Load pre-trained model  |

### To be added in table done
 

### Need to exclude from FAME-ML 
- Generator=G.permute(0,2,3,1)  
- model = Graph()
- graph = VGG_16_graph() 
- output=np.minimum(np.maximum(Generator,0.0), 255.0)  
- cls = PointNetCls(k = 5) 
- out, _, _ = cls(sim_data)
- model=cascaded_model(D_m) 
- self.relu = nn.ReLU()
- num_inputs = env.observation_space.shape[0] 
- num_outputs = env.action_space.shape[0]
- configs['model'] = pipeline_config.model
- label = os.path.basename(os.path.dirname(one_file)) 
- labels = [sent[3] for sent in input_batch_list]
- model.summary() 
- c_mat = confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)])
- f1 = f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)])
- f1_macro = f1_score(y_test, y_predict, average='macro') 
- acc = accuracy_score(y_test, y_predict)
- classification_loss = classification_loss( truth=truth, predicted=predicted, weights=weights, is_one_hot=True)
- batch_size = data.HP_batch_size 
- visdom_logger.load_previous_values(state.epoch, state.results)  
- ref = CaffeFunction('VGG_ILSVRC_19_layers.caffemodel') 