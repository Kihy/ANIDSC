from models.base_model import *
from .base_pipeline import *

class ScalerTrainer(BasePipeline):
    def __init__(
        self,
        batch_size=1024,
        train_split=0.8,
        steps=["train"],
        **kwargs,
    ):
        super().__init__(allowed=["model", "dataset_name", "files"],**kwargs)
        self.steps = steps
        self.batch_size = batch_size
        self.train_split=train_split
        self.training_size=0
        
    def train(self):
        for loader in self.train_data:
            for feature, label in tqdm(
                loader, desc=f"training {self.model.model_name} on {loader.dataset.name}", leave=False
            ):
                self.model.train_step(feature)
        self.model.save(self.dataset_name)
        
    def setup_dataset(self):
        self.train_data = []
        
        for f in self.files:
            dataset=load_dataset(
                    **f, percentage=[0.0, self.train_split], batch_size=self.batch_size
                )
            self.training_size+=len(dataset)
            self.train_data.append(dataset)

class OutlierDetectionTrainer(BasePipeline):
    def __init__(
        self,
        batch_size=1024,
        train_split=0.8,
        val_split=0.1,
        val_epochs=1,
        epochs=1,
        log_results=True,
        profile=False,
        steps=["train"],
        **kwargs,
    ):
        
        super().__init__(allowed=["model", "dataset_name", "files"],**kwargs)
        self.epochs=epochs
        self.steps = steps
        self.batch_size = batch_size
        self.train_split=train_split
        self.val_split=val_split
        self.training_size=0
        self.val_epochs=val_epochs
        self.log_results=log_results
        self.profile=profile
        
    def train(self):
        if self.log_results:
            self.writer=SummaryWriter(f'runs/{self.dataset_name}/{self.model.model_name}')
            
            if self.profile:
                prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'logs/{self.dataset_name}/{self.model.model_name}'),
                        record_shapes=False,
                        with_stack=False)
                prof.start()
                
        for i in tqdm(range(self.epochs)):
            self.model.on_train_begin()
            
            for loader in self.train_data:
                
                loss=[]
                start_time=time.time()
                for feature, label in tqdm(
                    loader, desc=f"training {self.model.model_name} on {loader.dataset.name}", leave=False
                ):
                    if self.profile:
                        prof.step()
                    loss.append(self.model.train_step(feature))

                if self.log_results:  
                    self.writer.add_scalar(f'training loss {loader.dataset.name}', np.mean(np.hstack(loss)), i)
                    self.writer.add_scalar(f'training time {loader.dataset.name}', time.time()-start_time, i) 
            
                loader.dataset.reset()
            self.model.on_train_end()
            
            if i % self.val_epochs==0 or i==self.epochs:
                self.calc_threshold(i)
                self.model.save(self.dataset_name ,i)

        if self.profile:
            prof.stop()  
            # self.writer.add_graph(self.model, torch.rand((128, self.model.n_features)).detach())
        
    def calc_threshold(self, i=0):
        # calculate threshold and records average loss
        scores = []
        for loader in tqdm(self.val_data):
            for feature, label in tqdm(loader, desc=f"calc threshold {self.model.model_name} on {loader.dataset.name}", leave=False):
                scores.append(self.model.predict_scores(feature))
                if self.log_results:
                    self.writer.add_scalar(f'val scores {loader.dataset.name}', np.mean(np.hstack(scores)), i)
            loader.dataset.reset()
        threshold = np.percentile(np.hstack(scores), 99.9)
        self.model.threshold=threshold
    
    
    def setup_dataset(self):
        self.train_data = []
        self.val_data = []
        
        for f in self.files:
            dataset=load_dataset(
                    **f, percentage=[0.0, self.train_split], batch_size=self.batch_size
                )
            self.training_size+=len(dataset)
            self.train_data.append(dataset)

            self.val_data.append(load_dataset(
                    **f,
                    percentage=[self.train_split, self.train_split + self.val_split],
                    batch_size=self.batch_size,
                ))
                

class OutlierDetectionEvaluator(BasePipeline):
    def __init__(
        self,
        batch_size=1024,
        plot=True,
        test_split=0.9,
        steps=["eval"],
        metrics=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metrics=metrics
        self.steps = steps
        self.batch_size = batch_size
        self.plot=plot
        self.test_split=test_split

    def plot_scores(self, scores, threshold, save_path):
        save_path=Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
            
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.scatter(range(len(scores)), scores, s=5)
        ax.axhline(threshold)
        ax.set_yscale('symlog', linthresh=2*threshold)
        idx=save_path.parts.index("plots")
        
        ax.set_title("-".join(save_path.parts[idx+1:]))
        fig.tight_layout()
        fig.savefig(save_path)
        print("save plot at", str(save_path))
        plt.close(fig)
    
    def eval(self):
        if isinstance(self.model, BaseDeepODModel):
            self.model.eval()
        for loader in self.test_data:
            scores = []
            
            start_time=time.time()
            for feature, label in tqdm(loader, desc=f"evaluating {loader.dataset.name}"):
                scores.append(self.model.predict_scores(feature))
            duration=time.time()-start_time
            
            scores=np.hstack(scores)
            
            if loader.dataset.file_name not in self.results[self.model.model_name]:
                self.results[self.model.model_name][loader.dataset.file_name]={}
            self.results[self.model.model_name][loader.dataset.file_name]["time"]=duration
                
            for metric in self.metrics:
                self.results[self.model.model_name][loader.dataset.file_name][metric.__name__]=metric(scores, self.model.threshold)
                
                if self.plot:
                    self.plot_scores(scores, self.model.threshold, f"../../datasets/{loader.dataset.dataset_name}/{loader.dataset.fe_name}/plots/{loader.dataset.file_name}/{self.model.model_name}.png")            
        
        with open(str(self.results_path),"w") as f:
            json.dump(self.results, f, indent=4)
               
    def setup_dataset(self):
        self.test_data=[]
        for f in self.files:
            if f["file_name"].startswith("benign"):
                self.test_data.append(load_dataset(
                        **f,
                        percentage=[self.test_split,1],
                        batch_size=self.batch_size,
                    ))
            else:
                self.test_data.append(load_dataset(
                        **f,
                        batch_size=self.batch_size,
                    ))

        self.results_path=Path(f"../../datasets/{self.dataset_name}/results.json")
        
        if self.results_path.is_file():
            self.results=json.load(open((str(self.results_path))))
            if self.model.model_name not in self.results.keys():
                self.results[self.model.model_name]={}
        else:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results={self.model.model_name:{}}


class OnlineODEvaluator(BasePipeline):
    def __init__(
        self,
        batch_size=1024,
        profile=False,
        steps=["process"],
        metrics=[],
        plot=True,
        percentage=[0,1],
        **kwargs,
    ):
        
        super().__init__(allowed=["model", "dataset_name", "files"],**kwargs)
        self.steps = steps
        self.batch_size = batch_size
        self.training_size=0
        self.profile=profile
        self.metrics=metrics
        self.plot=plot
        self.percentage=percentage
    
    def plot_scores(self, scores, save_path):
        save_path=Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
            
        fig, ax = plt.subplots(nrows=2, figsize=(6, 8))
        
        for name, value in scores.items():
            
            if name in ["detection_rate"]:
                ax[0].scatter(range(len(value)), value, s=5, label=name)
            if name in ["average_score","average_threshold"]:
                ax[1].scatter(range(len(value)), value, s=5, label=name)
        
        idx=save_path.parts.index("plots")
        
        fig.suptitle("-".join(save_path.parts[idx+1:]))
        ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax[1].set_yscale("log")
        fig.tight_layout()
        fig.savefig(save_path)
        print("save plot at", str(save_path))
        plt.close(fig)
    
    
    def process(self):
        for loader in self.datasets:
          
            #initialize file
            if loader.dataset.file_name not in self.results[self.model.model_name]:
                self.results[self.model.model_name][loader.dataset.file_name]={}
            
            # create list to store scores
            for metric in self.metrics:
                self.results[self.model.model_name][loader.dataset.file_name][metric.__name__]=[]
            
            start_time=time.time()
            count=0
            
            for feature, label in tqdm(
                loader, desc=f"processing {self.model.model_name} on {loader.dataset.name}", leave=False
            ):
                
                score, threshold = self.model.process(feature)

                for metric in self.metrics:
                    self.results[self.model.model_name][loader.dataset.file_name][metric.__name__].append(metric(score, threshold))
                
                
                count+=score.shape[0]
                
            duration=time.time()-start_time
            
            #record time and count
            
            self.results[self.model.model_name][loader.dataset.file_name]["time"]=duration
            self.results[self.model.model_name][loader.dataset.file_name]["count"]=count
            
            #plot metrics
            if self.plot:
                self.plot_scores(self.results[self.model.model_name][loader.dataset.file_name], f"../../datasets/{loader.dataset.dataset_name}/{loader.dataset.fe_name}/plots/{loader.dataset.file_name}/{self.model.model_name}.png")            
                
            #initialize save file
            scores_and_thresholds_path=Path(f"../../datasets/{loader.dataset.dataset_name}/{loader.dataset.fe_name}/outputs/{loader.dataset.file_name}/{self.model.model_name}.csv")
            scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(scores_and_thresholds_path),"w") as output_file:
                np.savetxt(output_file, np.array([self.results[self.model.model_name][loader.dataset.file_name]["average_score"],self.results[self.model.model_name][loader.dataset.file_name]["average_threshold"]]).T, delimiter=",")
            
            # measure average
            for metric in self.metrics:
                
                self.results[self.model.model_name][loader.dataset.file_name][metric.__name__]=np.mean(self.results[self.model.model_name][loader.dataset.file_name][metric.__name__])
                
            loader.dataset.reset()
            
        with open(str(self.results_path),"w") as f:
            json.dump(self.results, f, indent=4, cls=JSONEncoder)
            
    
    def save_model(self):
        self.model.save(self.dataset_name)        
    
    def setup_dataset(self):
        self.datasets = []
        
        for f in self.files:
            dataset=load_dataset(
                    **f, batch_size=self.batch_size,
                    percentage=self.percentage
                )
            self.datasets.append(dataset)


        self.results_path=Path(f"../../datasets/{self.dataset_name}/results.json")
        
        if self.results_path.is_file():
            self.results=json.load(open((str(self.results_path))))
            if self.model.model_name not in self.results.keys():
                self.results[self.model.model_name]={}
        else:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results={self.model.model_name:{}}
        
        
        