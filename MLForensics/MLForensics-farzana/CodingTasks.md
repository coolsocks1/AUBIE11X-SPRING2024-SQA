# After Discussion and Validation 


## Category-1: Data Loading

### Category-1.1: Data loading

#### Section-1.1.a 

> The following code elements can be extracted using the `https://github.com/paser-group/MLForensics/blob/190f2e23b305618b5dc38095973450b98ccd5858/FAME-ML/py_parser.py#L17` . First get the class name, then the attribute 
method name, then check arguments ... if the class name matches and the attribute method name matches, and no. of 
function arguments is greater than 0, then flag as a detection . Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection 

- torch.load(data.load_model_dir)
- data.load(data.dset_dir)
- pickle.load(fp)
- json.load(config_file)
- np.load(path)
- torch.load(checkpoint_path)
- latest_blob.download_to_filename(self.local_save_file)
- blob.upload_from_filename(self.local_save_file)
- visdom_logger.load_previous_values(state.epoch, state.results)
- coco_gt.loadRes(predictions=coco_predictions)
- yaml.load(f)
- hub.load(params.hub_module_url)
- data_loader_factory.get_data_loader(dataloader_params).load(input_context)
- tf.io.read_file(filename)
- tf.data.Dataset.from_tensor_slices(filenames)
- self.sp_model.Load(sp_model_file)
- TaggingDataLoader(data_config).load()
- pd.read_csv(attributes_file)
- tl.files.load_file_list() 
- ibrosa.load(filename, sr=sample_rate, mono=True) 
- data_utils.load_celebA(img_dim, image_data_format) 
- dset.MNIST(root=".", download=True) 
- tarfile.open(target_file)
- audio.load_wav(wav_path)
- Image.open(args.demo_image) 
- agent.replay_buffer.load(self.rbuf_filename) 
- h5py.File(hdf5_file, "a")

#### Section-1.1.b

> You will write a separate method in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following ... some code from `py.traverse.py` file will be helpful. Also, write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection 

- data = np.frombuffer(f.read(), np.uint8, offset=8)
- mnist_loader = get_loader(config),


#### Section-1.1.c 

> You will write a separate method in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations with >0 function arguments ... match exact string. Also, write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  

- load_randomly_augmented_audio(audio_path, self.sample_rate)
- _download(filename, working_directory, url_source)
- open(input_file,'r', encoding="utf8").readlines()
- open(args.wavenet_params, 'r')
- load(saver, sess, restore_from)
- load_generic_audio(self.audio_dir, self.sample_rate)
- load_audio(args.input_path)
- load_image_dataset(dset, img_dim, image_data_format) 
- download_from_url(path, url)
- get_raw_files(FLAGS.data_dir, _TEST_DATA_SOURCES)
- _load_vocab_file(vocab_file, reserved_tokens)
- load_attribute_dataset(args.attr_file) 
- read_h5file(os.path.join(os.getcwd(), 'train.h5')) 
- load_lua(args.input_t7) 




### Category-1.2: Model loading

#### Section-1.2.a 

> The following code elements can be extracted using the `https://github.com/paser-group/MLForensics/blob/190f2e23b305618b5dc38095973450b98ccd5858/FAME-ML/py_parser.py#L17` . First get the class name, then the attribute 
method name, then check arguments ... if the class name matches and the attribute method name matches, then flag as a detection. Match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection   

- DeepSpeech.load_model_package(package)
- tf.keras.models.load_model(model_weights_path) 
- model.load_state_dict(torch.load(data.load_model_dir)) 
- network.load_net() 
- vgg.load_from_npy_file() 
- caffe_parser.read_caffemodel() 
- tf.train.Checkpoint()
- tfhub.load()
- scipy.misc.imresize()


#### Section-1.2.b

> You will use the method you wrote in `1.1.b` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following ... some code from my `py.traverse.py` file  will be helpful. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection     

- model_dir_path = patch_path('models')
- ref = CaffeFunction('VGG_ILSVRC_19_layers.caffemodel') 

#### Section-1.2.c

> You will use the method you wrote in `1.1.c` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations with > 0 function arguments ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection    

- load_model(cls, path)
- load_decoder(labels, cfg= LMConfig)
- load_previous_values(self, start_epoch, results_state)
- load_pretrained(model, num_classes, settings)
- load_param(prefix, begin_epoch, convert=True)

#### Section-1.2.d

> You will new method in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations with > 0 function arguments ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection    

- model = SeqLabel(data), args1, auxs1 = load_checkpoint(prefix1, epoch1) 

## Category-2: Data  downloads 

#### Section-2.a

> The following code elements can be extracted using the `https://github.com/paser-group/MLForensics/blob/190f2e23b305618b5dc38095973450b98ccd5858/FAME-ML/py_parser.py#L17` . First get the class name, then the attribute 
method name, then check arguments ... if the class name matches and the attribute method name matches, then flag as a detection. Match exact string . Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection    


- wget.download('http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz')
- urllib.request.urlopen(request)
- model_zoo.load_url(url)
- urllib.urlretrieve() 
- agent.load(misc.download_model())

#### Section-2.b

> You will use the method you wrote in `1.1.c` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations with > 0 function arguments ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection    

- prepare_url_image(url)

## Category-3: Prediction Model

### Category-3.1: Model Input/ Model features

> You will write a separate method in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following . Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection   


- batch_size = data.HP_batch_size 

### Category-3.2: Label manipulation 

#### 3.2.a 

> You will use the method you wrote in `1.1.b` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations with > 0 function arguments ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  


- train_data, train_label = read_h5file(os.path.join(os.getcwd(), 'train.h5'))
- val_data, val_label = read_h5file(os.path.join(os.getcwd(), 'val.h5'))
- label = np.array(hf.get('label')) 
- label = load_image(f).convert('P') 
- label = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(self.nyud_dir, idx))['segmentation'].astype(np.uint8)
- label = os.path.basename(os.path.dirname(one_file)) 
- raw_data,raw_label = load_data_and_labels(fenci_right_save_path,fenci_wrong_save_path)
- label = hfw.create_dataset("labels", data=df_attr[list_col_labels].values)

#### 3.2.b 

> You will write a separate method in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following . Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  

- labels = [sent[3] for sent in input_batch_list]

### Category-3.3: Model Output

#### 3.3.a 
> You will use the method you wrote in `1.1.a` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  


- model.summary() 
- data.show_data_summary()



#### 3.3.b 
> You will use the method you wrote in `1.1.b` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations with > 0 function arguments ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  

- output_data = interpreter.get_tensor(output_details[0]['index'])
- pred_scores = evaluate(data, model, name, data.nbest)
- model = model.eval()

#### 3.3.c 
> You will use the method you wrote in `1.1.b` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  


- c_mat = confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)])
- f1 = f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)])
- f1_macro = f1_score(y_test, y_predict, average='macro') 
- acc = accuracy_score(y_test, y_predict)
- classification_loss = classification_loss( truth=truth, predicted=predicted, weights=weights, is_one_hot=True)



## Category-4: Data Pipelines

#### 4.1 

> You will use the method you wrote in `1.1.a` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  

- argparse.ArgumentParser(description='Input pipeline')

#### 4.2
> You will use the method you wrote in `1.1.b` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection 

- pipeline_config = pipeline_pb2.TrainEvalPipelineConfig() 


#### 4.3

> You will use the method you wrote in `1.1.c` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  
- get_configs_from_pipeline_file(FLAGS.pipeline_config_path)



#### 4.4 

> You will write a new method in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect variable assignments that look liek the following ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection 

- configs['model'] = pipeline_config.model

## Category-5: Reinforcement learning

### Category-5.1: Environment

> You will use the method you wrote in `1.1.a` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  


- inner_next_obs, inner_reward, done = wrapped_env.step(action)
- obs, reward, done, tt = env.step(action)
- state, reward, done, _ = env.step(action)
- obs, reward, done, info = env.step(action)
- next_state, reward, done, _ = env.step(action) 
- batch_ob, reward, done, info = self.env.step(action) 
- state, reward, done, _ = env.step(action)
- o, r, done, info = self.env.step(action) 
- policy = torch.load(args.model_path) 
- env = gym.make(params.env_name) 


> You will use the method developed in `4.4` inside  `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect variable assignments that look liek the following ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection 

- num_inputs = env.observation_space.shape[0] 
- num_outputs = env.action_space.shape[0]

### Category-5.2: State Observation

> You will use the method you wrote in `1.1.a` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this methods you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  

- obs, reward, done, tt = env.step(action) 


## Category-6: Classification decision of DNNs

### 6.1 
> You will write a new method in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this method you will detect library imports ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection  

- `from keras.models import *` 
- *import torch.nn* 
- *import torch*  
- *from keras.models import Sequential, model_from_json*
- *from keras.models import Model*

### 6.2 
> You will use your method developed in 1.1.b in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this method you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection:
- model.compile() 
- model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

- res = graph.predict({'input':X_check}, verbose=2)
- history = graph.fit({'input':X_train, 'output1':y_train, 'output2':y2_train}, nb_epoch=1000, verbose=2)
- logs = model.fit(dataset, epochs=1, steps_per_epoch=2)
- score = model_from_file.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
- self.relu = nn.ReLU()

> You will use your method developed in 1.1.b in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this method you will detect function declarations ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection:

- cls = PointNetCls(k = 5) 
- out, _, _ = cls(sim_data)
- model=cascaded_model(D_m) 
- G=model(D, label_images) 
- Generator=G.permute(0,2,3,1)  
- output=np.minimum(np.maximum(Generator,0.0), 255.0) 
- model = Model(inputs=[inputs], outputs=[conv10]) 
- model = Graph()
- graph = VGG_16_graph() 

### Category-7: Detect absence of Logging for Exception Handling 

> A lot of prior work has observed that exception handling is a common target of 
logging, and absence of logging for exception is considered as a negative deevlopment practice. We will detect where exceptions are printed but not logged. Exceptions can happen for many reasons and not all of them are necessary to log. But the excpetions that developers already are capturing using `catch` need to be checked to see if they use logging for them. 

> For this you will check `except` blocks and see what is inside the `except` block. If there is no logging then report an antipattern. 


### Category-8: Incomplete Logging 

> You will need your library import code from 6.1 to detect `import logging` and `from symnet.logger import logger` and `import tensorflow.compat.v1 as tf` and `import tensorflow as tf`. Then use your code from `1.1.a` to detect 
the following: 

- logging.getLogger().setLevel(logging.DEBUG) 
- logging.basicConfig() 
- logger.info('called with args\n{}'.format(pprint.pformat(vars(args)))) 
- tf.logging()
- tf.compat.v1.logging.info("Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size)) 

> Use your code from `1.1.a` in `https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/py_parser.py` to parse the following. For this method you will detect function declarations and assigned to a variable ... match exact string. Write a separate method in https://github.com/paser-group/MLForensics/blob/farzana/FAME-ML/lint_engine.py for detection:

- logger = logging.getLogger() #no mention of timestamp*

### Category-9: Detect absence of Logging 

> You will write a separate method that will report the presence (True) or absence (False) of logging in a Python file. The method will check for all the import-related statements and function declarations from `Category-7`. Detect presence for each of the imports and return it as a dictionary. If any of the import statement appears check for the presence of existence. Return presence or absence for the above-mentioned logging function declarations as a dictionary. If the input Python file has no logging import statements, then the dictionary will be empty. 

 
