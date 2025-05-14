import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

from config import *

class Poisson:
    def __init__(self, position, vitesse):
        self.position = np.array(position, dtype=float)
        self.vitesse = np.array(vitesse, dtype=float)
        
    def update_position(self, dt, zone_limite):
        self.position = self.position + self.vitesse * dt
        
        for i in range(3):
            if self.position[i] < -zone_limite or self.position[i] > zone_limite:
                self.vitesse[i] = -self.vitesse[i]
                self.position[i] = np.clip(self.position[i], -zone_limite, zone_limite)
    
class AleatoirePoissonBanc:
    def __init__(self, num_poisson=NUM_FISH, zone_limite=ZONE_LIMIT):

        self.poissons = []
        self.zone_limite = zone_limite
        
        for _ in range(num_poisson):
            position = np.random.uniform(-zone_limite*BOX_PROPORTIONS, zone_limite*BOX_PROPORTIONS, size=3)
            vitesse = np.random.uniform(-1, 1, size=3)
            vitesse = vitesse / np.linalg.norm(vitesse) * FISH_SPEED
            self.poissons.append(Poisson(position, vitesse))
    
    def update(self, dt=DT):
        for poisson in self.poissons:
            poisson.update_position(dt, self.zone_limite)
    
    def get_positions(self):
        return np.array([poisson.position for poisson in self.poissons])


def visualiser_mouvement_aleatoire(banc, num_frames=ANIMATION_FRAMES):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Limites de la visualisation
    zone_limite = banc.zone_limite
    ax.set_xlim([-zone_limite, zone_limite])
    ax.set_ylim([-zone_limite, zone_limite])
    ax.set_zlim([-zone_limite, zone_limite])
    
    # Configurer les étiquettes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Banc de Poissons - Mouvement Aléatoire')
    
    # Initialiser le scatter plot
    positions = banc.get_positions()
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='o',c=FISH_COLOR, s=MARKER_SIZE, alpha=0.8)


    def update(frame):
        banc.update()
        
        positions = banc.get_positions()
        
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        return scatter
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=ANIMATION_INTERVAL, blit=False)

    #sauvegarder l'animation
    if save_mp4:
        writer = FFMpegWriter(fps=FPS)
        ani.save('aleatoire.mp4', writer=writer)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    poisson_banc = AleatoirePoissonBanc()
    visualiser_mouvement_aleatoire(poisson_banc)