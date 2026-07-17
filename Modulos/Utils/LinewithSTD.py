import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

class PlotConfig:
    def __init__(self, x_label: str, y_label: str, x_ticks, y_ticks, legend: list[str], title_template: str):
        self.x_label = x_label
        self.y_label = y_label
        self.x_ticks = x_ticks
        self.y_ticks = y_ticks
        self.legend = legend
        self.title_template = title_template # Ex: "Analise: {file_name}"

class FileLoader:
    def __init__(self, folderA, folderB):
        self.folderA = Path(folderA)
        self.folderB = Path(folderB)
    
    def loadFilesDirs(self):
        # usando sorted para garantir que o par i da pasta A seja o mesmo da pasta B
        filesA = sorted(list(self.folderA.glob("*.csv")))
        filesB = sorted(list(self.folderB.glob("*.csv")))
        return filesA, filesB

class ColumnSelector:
    def __init__(self, filepath: str, mean_col: str, std_col: str, min_idx=None,max_idx=None):
        self.filepath   = filepath
        self.mean_col   = mean_col
        self.std_col    = std_col
        self.min_idx    = min_idx
        self.max_idx    = max_idx

    def run(self):
        df = pd.read_csv(self.filepath)
        print(f"DEBUG: Fatiando de {self.min_idx} até {self.max_idx}")
        df_sliced = df.iloc[self.min_idx : self.max_idx]

        
        # se a coluna de std não existir ou não for necessária, retornamos None
        std_data = df_sliced[self.std_col] if self.std_col in df_sliced.columns else None
        mean_data = df_sliced[self.mean_col]
        
        return mean_data, std_data
    
class LineFigure:
    def __init__(self, cfg: PlotConfig, save_path: str):
        self.cfg = cfg
        self.save_path = save_path

    def plot(self, mean1, mean2, std1=None, std2=None, show_std=True, title_context="",format="png"):
        fig, ax1 = plt.subplots(figsize=(18, 8)) 
        cmap = plt.get_cmap('tab10')

        # Configuração do Título Dinâmico
        ax1.set_title(title_context)
        ax1.set_xlabel(self.cfg.x_label)
        ax1.set_ylabel(self.cfg.y_label)
        ax1.set_xticks(self.cfg.x_ticks)
        ax1.set_yticks(self.cfg.y_ticks)
        ax1.set_ylim(0,1)
        ax1.tick_params(axis='x', rotation=45)

        
        ax1.plot(self.cfg.x_ticks, mean1, label=self.cfg.legend[0], color=cmap(0), linewidth=2)
        ax1.plot(self.cfg.x_ticks, mean2, label=self.cfg.legend[1], color=cmap(1), linewidth=2)

        
        if show_std and std1 is not None and std2 is not None:
            ax1.fill_between(self.cfg.x_ticks, mean1 - std1, mean1 + std1, color=cmap(0), alpha=0.2)
            ax1.fill_between(self.cfg.x_ticks, mean2 - std2, mean2 + std2, color=cmap(1), alpha=0.2)

        ax1.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self.save_path+ f".{format}",format=format,bbox_inches='tight')
        plt.close(fig) 

class PlottingPipeline:
    def __init__(self, folder_a, folder_b, save_base_dir, mean_col, std_col=None):
        self.loader = FileLoader(folder_a, folder_b)
        self.save_base_dir = Path(save_base_dir)
        self.save_base_dir.mkdir(parents=True, exist_ok=True)
        self.mean_col = mean_col
        self.std_col = std_col

    def run(self, cfg: PlotConfig, show_std: bool = True,min_idx:int = None, max_idx:int = None,format="png"):
        files_a, files_b = self.loader.loadFilesDirs()
        
        for i, (path_a, path_b) in enumerate(zip(files_a, files_b)):
            file_name = path_a.stem
            
            
            m1, s1 = ColumnSelector(path_a, self.mean_col, self.std_col,min_idx,max_idx).run()
            m2, s2 = ColumnSelector(path_b, self.mean_col, self.std_col,min_idx,max_idx).run()

            # 2. Geração do Título Personalizado
            # O template aceita {file_name} e {idx}
            full_title = cfg.title_template.format(file_name=file_name, idx=i)

            # 3. Plotagem
            save_path = self.save_base_dir / f"plot_{file_name}"
            fig_handler = LineFigure(cfg, str(save_path))
            
            fig_handler.plot(m1, m2, s1, s2, show_std=show_std, title_context=full_title,format=format)
            
            