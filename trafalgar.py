import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter, PillowWriter

from config import *

class Fish:
    def __init__(self, position, velocity, color=HEALTHY_COLOR):
        """
        Initialise un poisson avec une position et une vitesse en 3D.
        
        Args:
            position: Vecteur de position 3D [x, y, z]
            velocity: Vecteur de vitesse 3D [vx, vy, vz]
            color: Couleur du poisson (bleu par défaut, vert pour les contaminés)
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.contaminated = False  # Indique si le poisson est contaminé
        
    def update_position(self, dt, zone_limit):
        """
        Met à jour la position du poisson en fonction de sa vitesse et vérifie les limites.
        
        Args:
            dt: Intervalle de temps pour la mise à jour
            zone_limit: Limite de la zone (pour contenir les poissons)
        """
        # Mettre à jour la position selon l'équation: position(t + Δt) = position(t) + vitesse(t) × Δt
        self.position = self.position + self.velocity * dt
        
        # Vérifier les limites de la zone et inverser la direction si nécessaire
        for i in range(3):
            if self.position[i] < -zone_limit or self.position[i] > zone_limit:
                self.velocity[i] = -self.velocity[i]
                # Ajuster la position pour éviter de sortir de la zone
                self.position[i] = np.clip(self.position[i], -zone_limit, zone_limit)
    
    def set_contaminated(self, contaminator_velocity=None):
        """
        Marque le poisson comme contaminé et change sa couleur.
        Si une vitesse de contamination est fournie, adopte cette vitesse avec une légère variation.
        
        Args:
            contaminator_velocity: Vitesse du poisson contaminant (None pour le leader initial)
        """
        # Si le poisson est déjà contaminé, ne rien faire
        if self.contaminated:
            return
            
        # Marquer comme contaminé
        self.contaminated = True
        self.color = CONTAMINATED_COLOR
        
        # Si une vitesse de contamination est fournie, l'adopter avec une légère variation
        if contaminator_velocity is not None:
            # Copier la vitesse du contaminant
            self.velocity = np.copy(contaminator_velocity)
            
            # Ajouter une petite variation aléatoire
            variation = np.random.uniform(VARIATION_MIN, VARIATION_MAX, size=3)
            self.velocity += variation
            
            # Normaliser la vitesse pour maintenir une magnitude constante
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * FISH_SPEED
    
    def is_contaminated(self):
        """
        Vérifie si le poisson est contaminé.
        
        Returns:
            Boolean indiquant si le poisson est contaminé
        """
        return self.contaminated


class TrafalgarFishSchool:
    def __init__(self, num_fish=NUM_FISH, zone_limit=ZONE_LIMIT, contamination_distance=CONTAMINATION_DIST):
        """
        Initialise un banc de poissons avec l'effet Trafalgar.
        
        Args:
            num_fish: Nombre de poissons dans le banc
            zone_limit: Limite de la zone où les poissons peuvent nager
            contamination_distance: Distance maximale pour la propagation du comportement
        """
        self.fishes = []
        self.zone_limit = zone_limit
        self.contamination_distance = contamination_distance
        self.leader_index = None
        
        # Créer des poissons avec des positions aléatoires dans tout le volume d'eau
        for _ in range(num_fish):
            position = np.random.uniform(-zone_limit*BOX_PROPORTIONS, zone_limit*BOX_PROPORTIONS, size=3)
            velocity = np.random.uniform(-1, 1, size=3)
            # Normaliser la vitesse initiale
            velocity = velocity / np.linalg.norm(velocity) * FISH_SPEED
            self.fishes.append(Fish(position, velocity))
        
        # Choisir un leader initial et le marquer comme contaminé
        self.select_leader()
    
    def select_leader(self):
        """
        Sélectionne un poisson leader aléatoirement et le marque comme contaminé.
        """
        # Sélectionner un leader
        self.leader_index = np.random.randint(0, len(self.fishes))
        leader = self.fishes[self.leader_index]
        
        # Donner au leader une variation de vitesse
        leader_velocity_change = np.random.uniform(LEADER_VARIATION_MIN, LEADER_VARIATION_MAX, size=3)
        leader.velocity += leader_velocity_change
        
        # Normaliser la vitesse
        leader.velocity = leader.velocity / np.linalg.norm(leader.velocity) * FISH_SPEED
        
        # Marquer le leader comme contaminé
        leader.set_contaminated()
    
    def calculate_distances(self):
        """
        Calcule les distances entre tous les poissons.
        
        Returns:
            Matrice de distances [n_fish, n_fish]
        """
        positions = np.array([fish.position for fish in self.fishes])
        n_fish = len(self.fishes)
        distances = np.zeros((n_fish, n_fish))
        
        # Calculer la distance euclidienne entre chaque paire de poissons
        for i in range(n_fish):
            for j in range(i+1, n_fish):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def propagate_behavior(self):
        """
        Propage le comportement (contamination) des poissons contaminés aux voisins proches.
        Quand un poisson sain rencontre un poisson contaminé, il adopte la même vitesse
        que le poisson qui l'a contaminé (avec une petite variation) et devient contaminé à son tour.
        Un poisson contaminé ne peut pas être influencé par d'autres poissons.
        """
        # Calculer les distances entre tous les poissons
        distances = self.calculate_distances()
        
        # Liste des poissons contaminés avant cette propagation
        contaminated_indices = [i for i, fish in enumerate(self.fishes) if fish.is_contaminated()]
        
        # Mémoriser les poissons qui vont être contaminés pendant ce cycle
        new_contaminations = []
        
        # Pour chaque poisson non contaminé, vérifier s'il est proche d'un poisson contaminé
        for i, fish in enumerate(self.fishes):
            if not fish.is_contaminated():  # Ne traiter que les poissons sains
                # Vérifier s'il est proche d'un poisson contaminé
                for j in contaminated_indices:
                    if distances[i, j] < self.contamination_distance:
                        # Mémoriser ce poisson et celui qui l'a contaminé
                        new_contaminations.append((i, j))
                        break
        
        # Appliquer les nouvelles contaminations
        for i, j in new_contaminations:
            # Le poisson sain adopte la vitesse du contaminant avec une variation aléatoire
            self.fishes[i].set_contaminated(self.fishes[j].velocity)
    
    def update(self, dt=DT):
        """
        Met à jour la position de tous les poissons et propage le comportement.
        
        Args:
            dt: Intervalle de temps pour la mise à jour
        """
        # Propager le comportement à chaque cycle
        self.propagate_behavior()
        
        # Mettre à jour la position de tous les poissons
        for fish in self.fishes:
            fish.update_position(dt, self.zone_limit)
    
    def get_positions(self):
        """
        Retourne les positions de tous les poissons.
        
        Returns:
            Tableau numpy de positions [n_fish, 3]
        """
        return np.array([fish.position for fish in self.fishes])
    
    def get_colors(self):
        """
        Retourne les couleurs de tous les poissons.
        
        Returns:
            Liste des couleurs
        """
        return [fish.color for fish in self.fishes]


def visualize_trafalgar_effect(school, num_frames=ANIMATION_FRAMES):
    """
    Visualise le banc de poissons avec l'effet Trafalgar.
    
    Args:
        school: Instance de TrafalgarFishSchool
        num_frames: Nombre d'images pour l'animation
    """
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
    ax.set_title('Banc de Poissons - Effet Trafalgar')
    
    # Initialiser le scatter plot
    positions = school.get_positions()
    colors = school.get_colors()
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                         c=colors, marker='o', s=MARKER_SIZE, alpha=0.8)
    
    # Texte pour afficher le pourcentage de poissons contaminés
    contamination_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    
    # Tracer l'historique des positions du leader si activé
    leader_trail = None
    leader_positions = []
    if TRACK_LEADER:
        leader_positions.append(school.fishes[school.leader_index].position.copy())
        leader_trail = ax.plot(
            [leader_positions[0][0]], 
            [leader_positions[0][1]], 
            [leader_positions[0][2]], 
            'r-', alpha=0.5, linewidth=2
        )[0]
        if len(leader_positions) > 10:
            leader_positions.pop(0)
    
    def update(frame):
        # Mettre à jour la position des poissons
        school.update()
        
        # Mettre à jour le scatter plot
        positions = school.get_positions()
        colors = school.get_colors()
        
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        scatter.set_color(colors)
        
        # Calculer et afficher le pourcentage de poissons contaminés
        contaminated_count = sum(1 for fish in school.fishes if fish.is_contaminated())
        percentage = contaminated_count / len(school.fishes) * 100
        contamination_text.set_text(f"Contaminés: {percentage:.1f}%")
        
        # Mettre à jour la trace du leader si activée
        if TRACK_LEADER:
            leader_pos = school.fishes[school.leader_index].position.copy()
            leader_positions.append(leader_pos)
            leader_x = [pos[0] for pos in leader_positions]
            leader_y = [pos[1] for pos in leader_positions]
            leader_z = [pos[2] for pos in leader_positions]
            leader_trail.set_data(leader_x, leader_y)
            leader_trail.set_3d_properties(leader_z)
        
        return scatter, contamination_text
    
    # Créer l'animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=ANIMATION_INTERVAL, blit=False)

    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Moi'), bitrate=1800)
    ani.save('trafalgar.mp4', writer=writer)


if __name__ == "__main__":
    fish_school = TrafalgarFishSchool()
    visualize_trafalgar_effect(fish_school)