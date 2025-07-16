from src.env.object_env import species
import numpy as np

class Species(species.Species):
    def __init__(self, parents_genes=None, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.vary_possibility = 0.01
        self.genes = self.get_genes(parents_genes)
        self.speed = np.sum(self.genes["speed"])/2
        self.state["speed"] = self.speed
        self.state["life"] = (np.sum(self.genes["life"])/4)*100


    def step(self, *arg, **kwargs):
        self.state["hunger"] -= 0.02
        self.state["life"] -= 1
        super().step(*arg, **kwargs)
        if self.state["life"] <= 0:
            self.state["alive"] = False
    
    def get_genes(self,parents_genes=None):
        if parents_genes is not None:
            parents_genes[0] = self.vary_genes(parents_genes[0])
            parents_genes[1] = self.vary_genes(parents_genes[1])
            genes = {
                "speed": np.clip(np.sum(np.array([parents_genes[0]["speed"] , parents_genes[1]["speed"]]),axis=0), a_min=0, a_max=1),
                "life": np.clip(np.sum(np.array([parents_genes[0]["life"] , parents_genes[1]["life"]]),axis=0), a_min=0, a_max=1)
                }
        else:
            genes = {
                "speed": [np.random.choice([0,1], p=[0.8,0.2]) for _ in range(12)],
                "life": [np.random.choice([0,1], p=[0.8,0.2]) for _ in range(12)]
            }
        return genes
    
    def change_genes(self,gene):
        if self.vary_possibility >= np.random.rand():
            for i in [0,1]:
                if gene != i:
                    gene = i
        else:
            gene = gene           
        return gene

    def vary_genes(self, genes):
        diction = {}
        for key, value in genes.items():
            diction[key] = [self.change_genes(i) for i in value]
        return diction
