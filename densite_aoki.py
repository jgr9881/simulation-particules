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

class AokiTopologicalFishSchool:
    def __init__(self, num_fish=300, zone_limit=ZONE_LIMIT):
        self.num_fish = num_fish
        self.zone_limit = zone_limit
        self.fishes = []
        for _ in range(num_fish):
            pos = np.random.uniform(-zone_limit*BOX_PROPORTIONS,
                                     zone_limit*BOX_PROPORTIONS, size=3)
            vel = np.random.normal(size=3)
            vel /= np.linalg.norm(vel)
            vel *= FISH_SPEED
            self.fishes.append(Fish(pos, vel))

    def update(self, dt=DT):
        positions = np.array([f.position for f in self.fishes])
        tree = KDTree(positions)
        new_velocities = []

        for i, fish in enumerate(self.fishes):
            pos_i = fish.position
            vel_i = fish.velocity
            # Sélection topologique : K plus proches voisins (excl. soi-même)
            dists, idxs = tree.query(pos_i, k=NEAREST_NEIGHBORS+1)
            neighbors = [j for j in idxs if j != i]

            F_rep = np.zeros(3)
            F_ali = np.zeros(3)
            F_att = np.zeros(3)
            count_ali = 0

            for j in neighbors:
                nb = self.fishes[j]
                diff = nb.position - pos_i
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                # zones métriques sur voisins topologiques
                if dist < R_REPULSION:
                    F_rep += -K_REPULSION * (diff / dist)
                elif dist < R_ALIGNMENT:
                    F_ali += nb.velocity
                    count_ali += 1
                else:
                    F_att += K_ATTRACTION * (diff / dist)

            if count_ali > 0:
                F_ali /= count_ali

            v_new = vel_i + F_rep + F_ali + F_att
            norm = np.linalg.norm(v_new)
            if norm > 1e-6:
                v_new = v_new / norm * FISH_SPEED
            else:
                v_new = vel_i
            new_velocities.append(v_new)

        for fish, v_new in zip(self.fishes, new_velocities):
            fish.velocity = v_new
            fish.update_position(dt, self.zone_limit)

# Visualisation identique à Aoki métrique

def visualize_topological_school(school, frames=500, interval=50, save_mp4=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-school.zone_limit, school.zone_limit)
    ax.set_ylim(-school.zone_limit, school.zone_limit)
    ax.set_zlim(-school.zone_limit, school.zone_limit)
    ax.set_title(f'Aoki Topologique ({NEAREST_NEIGHBORS} voisins)')

    scatter = ax.scatter([], [], [], s=MARKER_SIZE, alpha=0.7)

    def init():
        scatter._offsets3d = ([], [], [])
        return scatter,

    def update_frame(frame):
        school.update(dt=DT)
        pos = np.array([f.position for f in school.fishes])
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        return scatter,

    ani = FuncAnimation(fig, update_frame, init_func=init,
                        frames=frames, interval=interval, blit=False)
    if save_mp4:
        writer = FFMpegWriter(fps=60)
        ani.save('aoki_densite.mp4', writer=writer)
    plt.show()

if __name__ == '__main__':
    school_topo = AokiTopologicalFishSchool(num_fish=NUM_FISH)
    visualize_topological_school(school_topo, save_mp4=True, frames=FRAMES)
