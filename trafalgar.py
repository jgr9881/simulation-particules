import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter, PillowWriter

from config import *

class Poisson:
    def __init__(self, position, vitesse, couleur=HEALTHY_COLOR):
        self.position = np.array(position, dtype=float)
        self.vitesse = np.array(vitesse, dtype=float)
        self.couleur = couleur
        self.contamine = False
        
    def update_position(self, dt, zone_limit):
        # position(t + Δt) = position(t) + vitesse(t) × Δt
        self.position = self.position + self.vitesse * dt
        
        for i in range(3):
            if self.position[i] < -zone_limit or self.position[i] > zone_limit:
                self.vitesse[i] = -self.vitesse[i]
                self.position[i] = np.clip(self.position[i], -zone_limit, zone_limit)
    
    def set_contamine(self, contaminateur_vitesse=None):
        if self.contamine:
            return
            
        self.contamine = True
        self.couleur = CONTAMINATED_COLOR
        
        if contaminateur_vitesse is not None:
            self.vitesse = np.copy(contaminateur_vitesse)
            
            variation = np.random.uniform(VARIATION_MIN, VARIATION_MAX, size=3)
            self.vitesse += variation
            
            self.vitesse = self.vitesse / np.linalg.norm(self.vitesse) * FISH_SPEED
    
    def is_contamine(self):
        return self.contamine


class BancPoissonsTrafalgar:
    def __init__(self, num_poisson=NUM_FISH, zone_limit=ZONE_LIMIT, contamination_distance=CONTAMINATION_DIST):
        self.poissones = []
        self.zone_limit = zone_limit
        self.contamination_distance = contamination_distance
        self.leader_index = None
        
        for _ in range(num_poisson):
            position = np.random.uniform(-zone_limit*BOX_PROPORTIONS, zone_limit*BOX_PROPORTIONS, size=3)
            vitesse = np.random.uniform(-1, 1, size=3)
            vitesse = vitesse / np.linalg.norm(vitesse) * FISH_SPEED
            self.poissones.append(Poisson(position, vitesse))
        
        self.select_leader()
    
    def select_leader(self):
        self.leader_index = np.random.randint(0, len(self.poissones))
        leader = self.poissones[self.leader_index]
        
        variation_vitesse_leader = np.random.uniform(LEADER_VARIATION_MIN, LEADER_VARIATION_MAX, size=3)
        leader.vitesse += variation_vitesse_leader
        
        leader.vitesse = leader.vitesse / np.linalg.norm(leader.vitesse) * FISH_SPEED
        
        leader.set_contamine()
    
    def calculate_distances(self):
        positions = np.array([poisson.position for poisson in self.poissones])
        n_poisson = len(self.poissones)
        distances = np.zeros((n_poisson, n_poisson))
        
        for i in range(n_poisson):
            for j in range(i+1, n_poisson):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def propagate_behavior(self):
        distances = self.calculate_distances()
        
        contamine_indices = [i for i, poisson in enumerate(self.poissones) if poisson.is_contamine()]
        
        new_contaminations = []
        for i, poisson in enumerate(self.poissones):
            if not poisson.is_contamine():
                for j in contamine_indices:
                    if distances[i, j] < self.contamination_distance:
                        new_contaminations.append((i, j))
                        break
        
        for i, j in new_contaminations:
            self.poissones[i].set_contamine(self.poissones[j].vitesse)
    
    def update(self, dt=DT):
        self.propagate_behavior()
        for poisson in self.poissones:
            poisson.update_position(dt, self.zone_limit)
    
    def get_positions(self):
        return np.array([poisson.position for poisson in self.poissones])
    
    def get_couleurs(self):
        return [poisson.couleur for poisson in self.poissones]


def visualiser_effet_trafalgar(banc, num_frames=ANIMATION_FRAMES):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    zone_limit = banc.zone_limit
    ax.set_xlim([-zone_limit, zone_limit])
    ax.set_ylim([-zone_limit, zone_limit])
    ax.set_zlim([-zone_limit, zone_limit])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Banc de Poissons - Effet Trafalgar')
    
    positions = banc.get_positions()
    couleurs = banc.get_couleurs()
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                         c=couleurs, marker='o', s=MARKER_SIZE, alpha=0.8)
    
    def update(frame):
        banc.update()
        
        positions = banc.get_positions()
        couleurs = banc.get_couleurs()
        
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        scatter.set_color(couleurs)
        
        return scatter
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=ANIMATION_INTERVAL, blit=False)

    if save_mp4:
        writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Moi'), bitrate=1800)
        ani.save('trafalgar.mp4', writer=writer)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    banc_de_poissons = BancPoissonsTrafalgar()
    visualiser_effet_trafalgar(banc_de_poissons)