import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation, FFMpegWriter

from config import *

class Fish:
    def __init__(self, position, velocity, color='blue'):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color

    def update_position(self, dt, zone_limit):
        self.position += self.velocity * dt
        # Rebond aux limites
        for i in range(3):
            if self.position[i] < -zone_limit or self.position[i] > zone_limit:
                self.velocity[i] = -self.velocity[i]
                self.position[i] = np.clip(self.position[i], -zone_limit, zone_limit)

class AokiFishSchool:
    def __init__(self, num_fish=NUM_FISH, zone_limit=ZONE_LIMIT):
        self.num_fish = num_fish
        self.zone_limit = zone_limit
        self.fishes = []
        # Initialisation des poissons
        for _ in range(num_fish):
            pos = np.random.uniform(-zone_limit*BOX_PROPORTIONS,
                                     zone_limit*BOX_PROPORTIONS, size=3)
            vel_dir = np.random.uniform(-1, 1, size=3)
            vel_dir /= np.linalg.norm(vel_dir)
            vel = vel_dir * FISH_SPEED
            self.fishes.append(Fish(pos, vel))

    def update(self, dt=DT):
        # Construire KDTree sur les positions actuelles
        positions = np.array([f.position for f in self.fishes])
        tree = KDTree(positions)

        new_velocities = []
        for i, fish in enumerate(self.fishes):
            pos_i = fish.position
            vel_i = fish.velocity

            # Chercher voisins dans le rayon d'attraction
            idxs = tree.query_ball_point(pos_i, r=R_ATTRACTION)
            # Exclure soi-même
            idxs = [j for j in idxs if j != i]

            # Initialisation des forces
            F_rep = np.zeros(3)
            F_ali = np.zeros(3)
            F_att = np.zeros(3)
            count_ali = 0

            # Parcourir les voisins
            for j in idxs:
                neighbor = self.fishes[j]
                d_vec = neighbor.position - pos_i
                dist = np.linalg.norm(d_vec)
                if dist < 1e-6:
                    continue
                # Zone de répulsion
                if dist < R_REPULSION:
                    F_rep += -K_REPULSION * (d_vec / dist)
                # Zone d'alignement
                elif dist < R_ALIGNMENT:
                    F_ali += neighbor.velocity
                    count_ali += 1
                # Zone d'attraction
                else:
                    F_att += K_ATTRACTION * (d_vec / dist)

            # Moyenne de l'alignement
            if count_ali > 0:
                F_ali = F_ali / count_ali
            # Calcul de la nouvelle vitesse
            v_new = vel_i + F_rep + F_ali + F_att
            # Normalisation à FISH_SPEED
            norm = np.linalg.norm(v_new)
            if norm > 1e-6:
                v_new = v_new / norm * FISH_SPEED
            else:
                # Si anomalie, conserver l'ancienne direction
                v_new = vel_i

            new_velocities.append(v_new)

        # Mise à jour des vitesses et positions
        for fish, v_new in zip(self.fishes, new_velocities):
            fish.velocity = v_new
            fish.update_position(dt, self.zone_limit)

    def get_positions(self):
        return np.array([f.position for f in self.fishes])

    def get_velocities(self):
        return np.array([f.velocity for f in self.fishes])

# Exemple de visualisation 3D pour Aoki

def visualize_aoki_school(school, frames=1000, interval=50, save_mp4=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-school.zone_limit, school.zone_limit)
    ax.set_ylim(-school.zone_limit, school.zone_limit)
    ax.set_zlim(-school.zone_limit, school.zone_limit)
    ax.set_title('Banc de Poissons - Modèle de Aoki')

    scatter = ax.scatter([], [], [], s=MARKER_SIZE, alpha=0.7)

    def init():
        scatter._offsets3d = ([], [], [])
        return scatter,

    def update(frame):
        school.update()
        pos = school.get_positions()
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        return scatter,

    ani = FuncAnimation(fig, update, init_func=init,
                        frames=frames, interval=interval, blit=False)
    if save_mp4:
        writer = FFMpegWriter(fps=FPS)
        ani.save('aoki_school.mp4', writer=writer)

    plt.show()

if __name__ == '__main__':
    school = AokiFishSchool()
    visualize_aoki_school(school, save_mp4=False, frames=FRAMES)
