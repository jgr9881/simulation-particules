import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation, FFMpegWriter

from config import *

class Poisson:
    def __init__(self, position, vitesse, couleur='blue'):
        self.position = np.array(position, dtype=float)
        self.vitesse = np.array(vitesse, dtype=float)
        self.couleur = couleur

    def update_position(self, dt, limite_zone):
        self.position += self.vitesse * dt
        for i in range(3):
            if self.position[i] < -limite_zone or self.position[i] > limite_zone:
                self.vitesse[i] = -self.vitesse[i]
                self.position[i] = np.clip(self.position[i], -limite_zone, limite_zone)

class BancPoissonsAokiDensite:
    def __init__(self, nb_poisson=300, limite_zone=ZONE_LIMIT):
        self.nb_poisson = nb_poisson
        self.limite_zone = limite_zone
        self.poissons = []
        for _ in range(nb_poisson):
            pos = np.random.uniform(-limite_zone*BOX_PROPORTIONS,
                                     limite_zone*BOX_PROPORTIONS, size=3)
            vel = np.random.normal(size=3)
            vel /= np.linalg.norm(vel)
            vel *= FISH_SPEED
            self.poissons.append(Poisson(pos, vel))

    def update(self, dt=DT):
        positions = np.array([f.position for f in self.poissons])
        tree = KDTree(positions)
        new_vitesses = []

        for i, poisson in enumerate(self.poissons):
            pos_i = poisson.position
            vel_i = poisson.vitesse
            dists, idxs = tree.query(pos_i, k=NEAREST_NEIGHBORS+1)
            voisins = [j for j in idxs if j != i]

            F_rep = np.zeros(3)
            F_ali = np.zeros(3)
            F_att = np.zeros(3)
            count_ali = 0

            for j in voisins:
                nb = self.poissons[j]
                diff = nb.position - pos_i
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue
                if dist < R_REPULSION:
                    F_rep += -K_REPULSION * (diff / dist)
                elif dist < R_ALIGNMENT:
                    F_ali += nb.vitesse
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
            new_vitesses.append(v_new)

        for poisson, v_new in zip(self.poissons, new_vitesses):
            poisson.vitesse = v_new
            poisson.update_position(dt, self.limite_zone)

def visualiser_aoki_densite(poisson, frames=500, interval=50, save_mp4=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-poisson.limite_zone, poisson.limite_zone)
    ax.set_ylim(-poisson.limite_zone, poisson.limite_zone)
    ax.set_zlim(-poisson.limite_zone, poisson.limite_zone)
    ax.set_title(f'Aoki ({NEAREST_NEIGHBORS} voisins)')

    scatter = ax.scatter([], [], [], s=MARKER_SIZE, alpha=0.7, c=FISH_COLOR)

    def init():
        scatter._offsets3d = ([], [], [])
        return scatter,

    def update_frame(frame):
        poisson.update(dt=DT)
        pos = np.array([f.position for f in poisson.poissons])
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        return scatter,

    ani = FuncAnimation(fig, update_frame, init_func=init,
                        frames=frames, interval=interval, blit=False)
    if save_mp4:
        writer = FFMpegWriter(fps=60)
        ani.save('aoki_densite.mp4', writer=writer)
    plt.show()

if __name__ == '__main__':
    poisson_densite = BancPoissonsAokiDensite(nb_poisson=NUM_FISH)
    visualiser_aoki_densite(poisson_densite, save_mp4=save_mp4, frames=FRAMES)
