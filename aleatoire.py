import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

from config import *

class Fish:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        
    def update_position(self, dt, zone_limit):
        self.position = self.position + self.velocity * dt
        
        for i in range(3):
            if self.position[i] < -zone_limit or self.position[i] > zone_limit:
                self.velocity[i] = -self.velocity[i]
                self.position[i] = np.clip(self.position[i], -zone_limit, zone_limit)
    
class AleatoireFishSchool:
    def __init__(self, num_fish=NUM_FISH, zone_limit=ZONE_LIMIT):

        self.fishes = []
        self.zone_limit = zone_limit
        
        for _ in range(num_fish):
            position = np.random.uniform(-zone_limit*BOX_PROPORTIONS, zone_limit*BOX_PROPORTIONS, size=3)
            velocity = np.random.uniform(-1, 1, size=3)
            velocity = velocity / np.linalg.norm(velocity) * FISH_SPEED
            self.fishes.append(Fish(position, velocity))
    
    def update(self, dt=DT):
        for fish in self.fishes:
            fish.update_position(dt, self.zone_limit)
    
    def get_positions(self):
        return np.array([fish.position for fish in self.fishes])


def visualize_mouvement_aleatoire(school, num_frames=ANIMATION_FRAMES):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Limites de la visualisation
    zone_limit = school.zone_limit
    ax.set_xlim([-zone_limit, zone_limit])
    ax.set_ylim([-zone_limit, zone_limit])
    ax.set_zlim([-zone_limit, zone_limit])
    
    # Configurer les étiquettes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Banc de Poissons - Mouvement Aléatoire')
    
    # Initialiser le scatter plot
    positions = school.get_positions()
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='o',c=FISH_COLOR, s=MARKER_SIZE, alpha=0.8)


    def update(frame):
        school.update()
        
        positions = school.get_positions()
        
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
    fish_school = AleatoireFishSchool()
    visualize_mouvement_aleatoire(fish_school)