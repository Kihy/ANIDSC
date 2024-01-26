from .base_adversarial_attack import BaseAdversarialAttack
from scapy.all import *
from tqdm import tqdm
from pyswarms.backend.topology import Topology
import logging
import numpy as np
from pyswarms.utils.reporter import Reporter
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.backend import operators as ops
from pyswarms.backend.swarms import Swarm
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import cKDTree
from pathlib import Path


def plot_contour(
    pos_history,
    canvas=None,
    title="Trajectory",
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
    original_time=0,
    **kwargs,
):
    """Draw a 2D contour map for particle trajectories
    Here, the space is represented as a flat plane. The contours indicate the
    elevation with respect to the objective function. This works best with
    2-dimensional swarms with their fitness in z-space.
    Parameters
    ----------
    pos_history : numpy.ndarray or list
        Position history of the swarm with shape
        :code:`(iteration, n_particles, dimensions)`
    canvas : (:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`),
        The (figure, axis) where all the events will be draw. If :code:`None`
        is supplied, then plot will be drawn to a fresh set of canvas.
    title : str, optional
        The title of the plotted graph. Default is `Trajectory`
    mark : tuple, optional
        Marks a particular point with a red crossmark. Useful for marking
        the optima.
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    mesher : :obj:`pyswarms.utils.formatters.Mesher`, optional
        Mesher class for mesh plots
    animator : :obj:`pyswarms.utils.formatters.Animator`, optional
        Animator class for custom animation
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :obj:`matplotlib.axes.Axes` plotting function
    Returns
    -------
    :obj:`matplotlib.animation.FuncAnimation`
        The drawn animation that can be saved to mp4 or other
        third-party tools
    """

    try:
        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(limits=[(-1, 1), (-1, 1)], label=["x-axis", "y-axis"])

        # If no Animator class supplied, use defaults
        if animator is None:
            animator = Animator()

        # If ax is default, then create new plot. Set-up the figure, the
        # axis, and the plot element that we want to animate
        if canvas is None:
            fig, ax = plt.subplots(1, 1, figsize=designer.figsize)
        else:
            fig, ax = canvas

        # Get number of iterations
        n_iters = len(pos_history)

        # Customize plot
        ax.set_title(title, fontsize=designer.title_fontsize)
        ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
        ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
        ax.set_xlim(designer.limits[0])
        ax.set_ylim(designer.limits[1])

        # Make a contour map if possible
        if mesher is not None:
            (
                xx,
                yy,
                zz,
            ) = _mesh(mesher)
            ax.contour(xx, yy, zz, levels=mesher.levels)

        # Mark global best if possible
        if mark is not None:
            ax.scatter(mark[0], mark[1], color="red", marker="x")

        n_particles = pos_history[0].shape[0]

        # last position is global best
        colours = ["#FF1E2E" for i in range(n_particles)]
        colours[-1] = "black"
        # Put scatter skeleton
        plot = ax.scatter(
            x=[0 for i in range(n_particles)],
            y=[0 for i in range(n_particles)],
            c=colours,
            zorder=1,
            alpha=0.6,
            **kwargs,
        )
        # Do animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate,
            frames=range(n_iters),
            fargs=(pos_history, plot, original_time, ax),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )
    except TypeError:
        rep.logger.exception("Please check your input type")
        raise
    else:
        return anim


def _animate(i, data, plot, original_time, ax):
    """Helper animation function that is called sequentially
    :class:`matplotlib.animation.FuncAnimation`
    """
    current_pos = data[i]
    current_pos[:, 0] -= original_time
    current_pos = current_pos[:, :2]
    if i is not 0:
        colours = ["#1EFFEF" for i in range(current_pos.shape[0])]
        colours[-1] = "black"
        ax.scatter(
            x=data[i - 1][:, 0],
            y=data[i - 1][:, 1],
            c=colours,
            alpha=0.2,
            zorder=-1,
            s=5,
        )

    ax.set_title("iteration:{}".format(i))
    if np.array(current_pos).shape[1] == 2:
        plot.set_offsets(current_pos)
    else:
        plot._offsets3d = current_pos.T
    return (plot,)


def boost_factor(i):
    """
    boosts the velocity if particle makes no improvement
    Args:
        i (int): number of iteration since last improvement.

    Returns:
        float: boosted velocity

    """
    # cap factor at 20 iterations
    i = np.where(i > 10, 10, i)
    i = np.expand_dims(i, axis=1)
    return np.exp(0.05 * i)


def create_swarm(n_particles, options, bounds):
    """
    initializes the swarm

    Args:
        n_particles (int): number of particles.
        options (dict): options to generate swarm.
        bounds (tuple): bounds for each dimension.

    Returns:
        swarm instance with position and velocity
    """

    position = generate_position(n_particles, bounds)
    velocity = generate_velocity(n_particles, bounds)

    swarm = Swarm(position, velocity, options=options)
    return swarm


def generate_velocity(n_particles, bounds):
    """
    generates velocity of each particle randomly

    Args:
        n_particles (int): number of particles.
        dimensions (int): dimensions of each particle.
        bounds (tuple): bounds for each dimension.

    Returns:
        array of velocities

    Raises:
        ExceptionName: Why the exception is raised.

    """
    lb, ub = bounds
    range = ub - lb
    factor = 0.2

    # set velocity to random value between 0 to factor of range
    velocity = np.random.uniform(0, factor * range, size=(n_particles, len(lb)))

    return velocity


def generate_position(n_particles, bounds):
    """
    randomly generates initial postions of swarm

    Args:
        n_particles (int): number of particles.
        dimensions (int): dimensions of each particle.
        bounds (tuple): bounds for each dimension.

    Returns:
        type: Description of returned object.

    Raises:
        ExceptionName: Why the exception is raised.

    """
    # assume discrete dimensions are all after continous ones
    lb, ub = bounds

    positions = np.random.uniform(low=lb, high=ub, size=(n_particles, len(lb)))

    # first position indicates no change
    positions[0] = lb

    return positions


class Traffic(Topology):
    def __init__(self, static=True):
        """
        initializes the topology

        Args:
            static (boolean): if true the topology dont change. Defaults to True.

        Returns:
            none

        """
        super(Traffic, self).__init__(static)
        self.rep = Reporter(logger=logging.getLogger(__name__))

        # contains information about randomly generated packets
        self.auxiliary_info = []
        self.bhn = BoundaryHandler(strategy="nearest")
        self.bhr = BoundaryHandler(strategy="random")

    def mutate_swarm(self, swarm, mutation_factor, crossp, mutation_candidates, bounds):
        """
        mutates the swarm with DE.

        Args:
            swarm (swarm): swarm instance.
            mutation_factor (float): hyperparameter to multiply with the difference.
            crossp (float): recombination probability.
            mutation_candidates (array): possible mutation candidate for each particle.
            bounds (tuple): bounds for dimensions.

        Returns:
            array: mutated swarm

        """

        # evaluation
        trial_pop = []
        positions = swarm.position
        n_particles = swarm.position.shape[0]
        n_dims = swarm.position.shape[1]
        # mutation
        for i in range(n_particles):
            mutation_candidate = np.random.choice(
                mutation_candidates[i], 3, replace=False
            )
            a, b, c = positions[mutation_candidate]
            mutation_factor = 0.8
            mutant = a + mutation_factor * (b - c)
            # recombination
            crossp = 0.7
            cross_points = np.random.rand(n_dims) < crossp
            trial = np.where(cross_points, mutant, positions[i])

            trial_pop.append(trial)
        trial_pop = np.array(trial_pop)
        # round and bound population
        if np.random.rand() < 0.5:
            trial_pop = self.bhr(trial_pop, bounds)
        else:
            trial_pop = self.bhn(trial_pop, bounds)

        return trial_pop

    def compute_gbest_local(self, swarm, p, k, **kwargs):
        """
        compute and update the neighbourhood best particle in PSO

        Args:
            swarm (swarm): swarm instance.
            p (int): p=1 l1 distance, p=2 l2 distance
            k (int): number of neighbours for each particle.

        Returns:
            best cost, position, and craft configuration

        """
        try:
            # Check if the topology is static or not and assign neighbors
            if (self.static and self.neighbor_idx is None) or not self.static:
                # Obtain the nearest-neighbors for each particle
                tree = cKDTree(swarm.position)
                # p=1 l1 distance, p=2 l2 distance
                _, self.neighbor_idx = tree.query(swarm.position, p=p, k=k)

            # Map the computed costs to the neighbour indices and take the
            # argmin. If k-neighbors is equal to 1, then the swarm acts
            # independently of each other.
            if k == 1:
                # The minimum index is itself, no mapping needed.
                self.neighbor_idx = self.neighbor_idx[:, np.newaxis]
                best_neighbor = np.arange(swarm.n_particles)
            else:
                idx_min = swarm.pbest_cost[self.neighbor_idx].argmin(axis=1)
                best_neighbor = self.neighbor_idx[
                    np.arange(len(self.neighbor_idx)), idx_min
                ]

            # Obtain best cost and position
            best_cost = swarm.pbest_cost[best_neighbor]
            best_pos = swarm.pbest_pos[best_neighbor]

            # update best
            mask_cost = best_cost < swarm.best_cost
            mask_pos = np.expand_dims(mask_cost, axis=1)
            best_cost = np.where(mask_cost, best_cost, swarm.best_cost)
            best_pos = np.where(mask_pos, best_pos, swarm.best_pos)
            best_index = np.argmin(best_cost)

        except AttributeError:
            self.rep.logger.exception(
                "Please pass a Swarm class. You passed {}".format(type(swarm))
            )
            raise
        else:
            return best_pos, best_cost, best_index

    def compute_gbest(self, swarm, **kwargs):
        """
        computes global best in the swarm

        Args:
            swarm (swarm): swarm instance.

        Returns:
            global best pos, cost, aux

        """
        if np.min(swarm.pbest_cost) < swarm.best_cost:
            # Get the particle position with the lowest pbest_cost
            # and assign it to be the best_pos
            min_index = np.argmin(swarm.pbest_cost)
            best_pos = swarm.pbest_pos[min_index]
            best_cost = swarm.pbest_cost[min_index]
            best_aux = swarm.pbest_aux[min_index]
        else:
            # Just get the previous best_pos and best_cost
            best_pos, best_cost, best_aux = (
                swarm.best_pos,
                swarm.best_cost,
                swarm.best_aux,
            )

        return best_pos, best_cost, best_aux

    def compute_mbest(self, swarm):
        """
        computes and updates the cost of original and mutated particle

        Args:
            swarm (swarm): swarm instance

        Returns:
            new pos, cost and aux

        """
        # compares between current and trial
        # Create a 1-D and 2-D mask based from comparisons
        mask_cost = swarm.trial_cost < swarm.current_cost
        mask_pos = np.expand_dims(mask_cost, axis=1)
        # Apply masks
        new_pos = np.where(~mask_pos, swarm.position, swarm.trial_pos)
        new_cost = np.where(~mask_cost, swarm.current_cost, swarm.trial_cost)

        # apply aux info
        new_aux = np.where(~mask_pos, swarm.current_aux, swarm.trial_aux)

        return (new_pos, new_cost, new_aux)

    def compute_pbest(self, swarm, iter):
        """
        computes and updates the cost of original and moved particle

        Args:
            swarm (swarm): swarm instance

        Returns:
            new pos, cost and aux

        """
        # Create a 1-D and 2-D mask based from comparisons
        mask_cost = swarm.current_cost < swarm.pbest_cost

        mask_pos = np.expand_dims(mask_cost, axis=1)
        # Apply masks
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
        new_pbest_cost = np.where(~mask_cost, swarm.pbest_cost, swarm.current_cost)

        # record the iteration with best cost
        new_pbest_iter = np.where(~mask_cost, swarm.pbest_iter, iter)

        return new_pbest_pos, new_pbest_cost, new_pbest_iter

    def compute_velocity(
        self,
        swarm,
        clamp=None,
        vh=VelocityHandler(strategy="unmodified"),
        bounds=None,
        iter=None,
    ):
        """Compute the velocity matrix
        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.
        A sample usage can be seen with the following:
        .. code-block :: python
            import pyswarms.backend as P
            from pyswarms.backend.swarm import Swarm
            from pyswarms.backend.handlers import VelocityHandler
            from pyswarms.backend.topology import Star
            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()
            my_vh = VelocityHandler(strategy="adjust")
            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp, my_vh,
                bounds)
        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh : pyswarms.backend.handlers.VelocityHandler
            a VelocityHandler instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]
        # Compute for cognitive and social terms
        cognitive = (
            c1
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.pbest_pos - swarm.position)
        )
        social = (
            c2 * np.random.uniform(0, 1, swarm_size) * (swarm.best_pos - swarm.position)
        )

        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (w * swarm.velocity) + cognitive + social

        # temp_velocity *= boost_factor(iter - swarm.pbest_iter)

        updated_velocity = vh(
            temp_velocity, clamp, position=swarm.position, bounds=bounds
        )

        return updated_velocity

    def compute_position(
        self, swarm, bounds=None, bh=BoundaryHandler(strategy="random")
    ):
        """Update the position matrix
        This method updates the position matrix given the current position and
        the velocity. If bounded, it waives updating the position.
        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh : pyswarms.backend.handlers.BoundaryHandler
            a BoundaryHandler instance
        Returns
        -------
        numpy.ndarray
            New position-matrix
        """

        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            temp_position = bh(temp_position, bounds)

        position = temp_position

        return position


class LiuerMihouAttack(BaseAdversarialAttack):
    def __init__(
        self,
        max_num_adv=100,
        bounds={"max_time_delay": 0.1, "max_craft_pkt": 5, "max_payload_size": 1514},
        pso={
            "n_particles": 30,
            "iterations": 20,
            "options": {"c1": 0.7, "c2": 0.3, "w": 0.5},
            "p": 2,
            "k": 4,
            "clamp": None,
        },
        **kwargs,
    ):
        self.max_num_adv = max_num_adv
        self.bounds = bounds
        self.pso = pso

        self.allowed = ("fe", "model", "mal_pcap")

        for k, v in kwargs.items():
            assert k in self.allowed
            setattr(self, k, v)

    def __rrshift__(self, other):
        self.start(**other)

    def attack_setup(self):
        self.input_pcap = PcapReader(self.mal_pcap)
        self.mal_pcap = Path(self.mal_pcap)

        adv_pcap = (
            self.mal_pcap.parents[1] / "adversarial" / (self.mal_pcap.stem + "_lm.pcap")
        )
        adv_pcap.parent.mkdir(parents=True, exist_ok=True)
        self.output_pcap = PcapWriter(str(adv_pcap))
        print(f"adverarial pcap at: {adv_pcap}")

        log_path = (
            self.mal_pcap.parents[1]
            / "adversarial"
            / (self.mal_pcap.stem + "_lm_log.txt")
        )
        self.log_file = open(log_path, "w")

    def attack_teardown(self):
        self.input_pcap.close()
        self.output_pcap.close()
        self.log_file.close()

    def start(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.allowed
            setattr(self, k, v)

        self.craft_adversary()

    def craft_adversary(self):
        self.attack_setup()

        pkt_index = 0
        offset_time = 0
        num_adv = 0

        pbar = tqdm(total=self.max_num_adv, position=0)
        for packet in self.input_pcap:
            if num_adv > self.max_num_adv:
                break

            traffic_vector = self.fe.get_traffic_vector(packet)

            # shift packet time
            traffic_vector[-2] += offset_time

            pkt_index += 1

            if traffic_vector == None:
                packet.time = traffic_vector[-2]
                self.output_pcap.write(packet)
                prev_pkt_time = packet.time
                continue

            # find original score
            features = self.fe.peek([traffic_vector])
            label = self.model.predict_labels(features)

            if not label:  # packet is benign
                packet.time = traffic_vector[-2]
                self.output_pcap.write(packet)
                features = self.fe.update(traffic_vector)

                prev_pkt_time = packet.time
                continue

            min_time = prev_pkt_time - traffic_vector[-2]
            min_payload = len(packet) - len(packet.payload)

            bounds = np.array(
                [
                    [min_time, 0.0, min_payload],
                    [
                        self.bounds["max_time_delay"],
                        self.bounds["max_craft_pkt"],
                        self.bounds["max_payload_size"],
                    ],
                ]
            )

            cost, pos = self.optimize(traffic_vector, bounds)

            delay, n_craft, payload = pos[0], int(pos[1]), int(pos[2])

            # apply fake packets to feature extractor
            delay_times = np.linspace(0, delay, n_craft + 1, endpoint=False)[1:]
            for i in range(n_craft):
                craft_vector = traffic_vector[:-2] + [
                    prev_pkt_time + delay_times[i],
                    payload,
                ]
                self.fe.update(craft_vector)
                craft_packet = self.packet_gen(
                    packet, craft_vector, modify_payload=True
                )
                self.output_pcap.write(craft_packet)

            traffic_vector[-2] = prev_pkt_time + delay
            self.fe.update(traffic_vector)
            adversarial_packet = self.packet_gen(
                packet, traffic_vector, modify_payload=False
            )
            self.output_pcap.write(adversarial_packet)
            prev_pkt_time = adversarial_packet.time

            num_adv += 1
            pbar.update(1)

        self.attack_teardown()

    def packet_gen(self, packet, traffic_vector, modify_payload=False):
        packet.time = traffic_vector[-2]

        if packet.haslayer(IP):
            packet[IP].src = traffic_vector[3]
            packet[IP].dst = traffic_vector[5]

        if packet.haslayer(TCP):
            packet[TCP].sport = int(traffic_vector[4])
            packet[TCP].dport = int(traffic_vector[6])
            if modify_payload:
                packet[TCP].remove_payload()
                payload_size = int(traffic_vector[-1]) - len(packet)
                packet[TCP].add_payload(Raw(load="a" * payload_size))

            del packet[TCP].chksum

        if packet.haslayer(UDP):
            packet[UDP].sport = int(traffic_vector[4])
            packet[UDP].dport = int(traffic_vector[6])
            if modify_payload:
                packet[UDP].remove_payload()
                payload_size = int(traffic_vector[-1]) - len(packet)
                packet[UDP].add_payload(Raw(load="a" * payload_size))

        if packet.haslayer(ARP):
            packet[ARP].hwsrc = traffic_vector[1]
            packet[ARP].hwdst = traffic_vector[2]

        if packet.haslayer(IP):
            del packet[IP].len
            del packet[IP].chksum
            del packet.len

        return packet

    def cost_function(self, x, traffic_vector):
        original_time = traffic_vector[-2]

        split_idx = [0]
        features = []
        counter = 0
        # iterative through each particle
        for time_delay, n_craft, packet_size in x:
            n_craft = int(n_craft)
            packet_size = int(packet_size)
            simulated_tv = []
            delay_times = np.linspace(0, time_delay, n_craft + 1, endpoint=False)[1:]

            for t in delay_times:
                simulated_tv.append(
                    traffic_vector[:-2] + [original_time + t, packet_size]
                )

            traffic_vector[-2] = original_time + time_delay
            simulated_tv.append(traffic_vector)

            craft_features = self.fe.peek(simulated_tv)
            features.append(craft_features)
            counter += len(craft_features)
            split_idx.append(counter)

        anomaly_scores = self.model.predict_scores(np.vstack(features))

        cost = []
        for i, j in zip(split_idx[:-1], split_idx[1:]):
            cost.append(np.max(anomaly_scores[i:j]))

        return np.array(cost)

    def optimize(self, traffic_vector, bounds):
        topology = Traffic()

        swarm = create_swarm(
            n_particles=self.pso["n_particles"],
            options=self.pso["options"],
            bounds=bounds,
        )

        pbar = tqdm(range(self.pso["iterations"]), position=1, leave=False)

        for i in pbar:
            # Part 1: Update personal best
            swarm.current_cost = self.cost_function(
                swarm.position, traffic_vector
            )  # Compute current cost
            if i == 0:
                swarm.pbest_cost = swarm.current_cost
                swarm.best_cost = swarm.current_cost
                swarm.best_pos = swarm.position
                swarm.pbest_iter = np.zeros((self.pso["n_particles"],))

            (
                swarm.pbest_pos,
                swarm.pbest_cost,
                swarm.pbest_iter,
            ) = topology.compute_pbest(
                swarm, i
            )  # Update and store

            # Part 2: Update global best
            # Note that gbest computation is dependent on your topology
            # if np.min(swarm.pbest_cost) < swarm.best_cost:
            # best index is global minimum, others are best in the neighbourhood
            (
                swarm.best_pos,
                swarm.best_cost,
                swarm.best_index,
            ) = topology.compute_gbest_local(swarm, self.pso["p"], self.pso["k"])

            # Part 3: Update position and velocity matrices
            # Note that position and velocity updates are dependent on your topology

            swarm.velocity = topology.compute_velocity(
                swarm, bounds=bounds, clamp=self.pso["clamp"], iter=i
            )
            if np.random.rand() < 0.5:
                strat = "random"
            else:
                strat = "nearest"

            swarm.position = topology.compute_position(
                swarm, bounds=bounds, bh=BoundaryHandler(strategy=strat)
            )

            post_fix = "c: {:.4f}".format(swarm.best_cost[swarm.best_index])
            pbar.set_postfix_str(post_fix)

        return swarm.best_cost[swarm.best_index], swarm.best_pos[swarm.best_index]
