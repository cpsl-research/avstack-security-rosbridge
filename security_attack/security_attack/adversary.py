import numpy as np
import rclpy
from avsec.multi_agent.adversary import AdversaryModel
from avsec.multi_agent.manifest import (
    FalseNegativeManifest,
    FalsePositiveManifest,
    TranslationManifest,
)
from avsec.multi_agent.propagation import (
    MarkovPropagator,
    StaticPropagator,
    TrajectoryPropagator,
)
from avstack_bridge.base import Bridge
from avstack_bridge.detections import DetectionBridge
from rclpy.duration import Duration
from rclpy.node import Node
from ros2node.api import get_node_names
from tf2_ros import TransformException, TransformListener
from tf2_ros.buffer import Buffer
from vision_msgs.msg import Detection3DArray


class AdversaryNode(Node):
    def __init__(self):
        super().__init__("adversary")

        # general attack parameters
        self.declare_parameter(name="debug", value=True)
        self.declare_parameter(name="attack_agent_name", value="agent0")

        # adversary parameters
        self.declare_parameter(name="adv_dt_init", value=1.0)
        self.declare_parameter(name="adv_dt_reset", value=10.0)

        # -- manifest
        self.declare_parameter(name="manifest_fp_poisson", value=2.0)
        self.declare_parameter(name="manifest_fn_fraction", value=0.20)
        self.declare_parameter(name="manifest_tr_fraction", value=0.0)

        # -- propagation
        self.declare_parameter(name="propagation", value="StaticPropagator")
        self.declare_parameter(name="markov_propagator_v_sigma", value=10.0)
        self.declare_parameter(name="markov_propagator_dv_sigma", value=1.0)
        self.declare_parameter(name="trajectory_propagator_dx", value=20.0)
        self.declare_parameter(name="trajectory_propagator_dy", value=5.0)
        self.declare_parameter(name="trajectory_propagator_dz", value=0.0)
        self.declare_parameter(name="trajectory_propagator_dt", value=10.0)

        # -- build model
        manifest_fp, manifest_fn, manifest_tr = self.get_manifests()
        self.model = AdversaryModel(
            dt_init=self.get_parameter("adv_dt_init").value,
            dt_reset=self.get_parameter("adv_dt_reset").value,
            propagator=self.get_propagator(),
            manifest_fp=manifest_fp,
            manifest_fn=manifest_fn,
            manifest_tr=manifest_tr,
        )

        # transform listener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # set up subscriber/publisher for detections
        self.in_subscriber = self.create_subscription(
            Detection3DArray, "input", self.receive_detections, 10
        )
        self.out_publisher = self.create_publisher(Detection3DArray, "output", 10)

    @property
    def name(self):
        return get_node_names(node=self)

    @property
    def debug(self):
        return self.get_parameter("debug").value

    def get_propagator(self):
        """Use the parameters to set up the propagation model"""
        prop_model = self.get_parameter("propagation").value
        if prop_model == "StaticPropagator":
            propagator = StaticPropagator()
        elif prop_model == "MarkovPropagator":
            propagator = MarkovPropagator(
                v_sigma=self.get_parameter("markov_propagator_v_sigma").value,
                dv_sigma=self.get_parameter("markov_propagator_dv_sigma").value,
            )
        elif prop_model == "TrajectoryPropagator":
            dx = np.array(
                [
                    self.get_parameter("trajectory_propagator_dx").value,
                    self.get_parameter("trajectory_propagator_dy").value,
                    self.get_parameter("trajectory_propagator_dz").value,
                ]
            )
            dt = self.get_parameter("trajectory_propagator_dt").value
            propagator = TrajectoryPropagator(dx_total=dx, dt_total=dt)
        else:
            raise NotImplementedError(prop_model)
        return propagator

    def get_manifests(self):
        """Use the parameters to set up the manifest model"""
        manifest_fp = FalsePositiveManifest(
            fp_poisson=self.get_parameter("manifest_fp_poisson").value,
        )
        manifest_fn = FalseNegativeManifest(
            fn_fraction=self.get_parameter("manifest_fn_fraction").value,
        )
        manifest_tr = TranslationManifest(
            tr_fraction=self.get_parameter("manifest_tr_fraction").value,
        )
        return manifest_fp, manifest_fn, manifest_tr

    async def receive_detections(self, msg: Detection3DArray):
        """Called when intercepting detections from the compromised agent/simulator"""
        if self.debug:
            self.get_logger().info(
                "Received {} detections at the adversary".format(len(msg.detections))
            )

        # get agent reference frame
        try:
            tf_world_agent = self._tf_buffer.lookup_transform(
                target_frame="world",
                source_frame=self.get_parameter("attack_agent_name").value,
                time=msg.header.stamp,  # get the latest pose,
                timeout=Duration(seconds=0, nanoseconds=100 * 1e6),
            )
        except TransformException:
            self.out_publisher.publish(msg)
            if self.debug:
                self.get_logger().info("Cannot get agent transform...")
        else:
            reference_agent = Bridge.tf2_to_reference(tf_world_agent)

            # run adversary to manipulate objects
            objects = DetectionBridge.detectionarray_to_avstack(msg)
            # HACK: manually set the reference frame
            for obj in objects:
                obj.box.position.reference = reference_agent
                obj.box.attitude.reference = reference_agent
            objects = self.model(
                objects=objects,
                reference_agent=reference_agent,
            )

            # send new outputs
            msg_out = DetectionBridge.avstack_to_detectionarray(
                objects, header=msg.header
            )
            self.out_publisher.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)

    adversary = AdversaryNode()

    rclpy.spin(adversary)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    adversary.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
