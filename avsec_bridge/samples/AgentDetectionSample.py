import numpy as np
import rclpy
from avstack.environment.objects import ObjectState
from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    Box3D,
    GlobalOrigin3D,
    Position,
    Velocity,
    q_stan_to_cam,
)
from avstack.modules.perception.detections import BoxDetection
from avstack_bridge import Bridge
from avstack_bridge.detections import DetectionBridge
from avstack_bridge.objects import ObjectStateBridge
from avstack_msgs.msg import ObjectStateArray
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from vision_msgs.msg import Detection3DArray


def get_object_global(seed, timestamp: float, reference=GlobalOrigin3D):
    np.random.seed(seed)
    vel = 1 * np.random.randn(3)
    pos = 10 * np.random.randn(3) + timestamp * vel
    vel[2] = 0
    pos[2] = 0

    pos_obj = Position(pos, reference)
    rot_obj = Attitude(q_stan_to_cam, reference)
    box_obj = Box3D(pos_obj, rot_obj, [2, 2, 5])  # box in local coordinates
    vel_obj = Velocity(vel, reference)
    acc_obj = Acceleration(np.random.rand(3), reference)
    ang_obj = AngularVelocity(np.quaternion(1), reference)
    obj = ObjectState("car")
    obj.set(timestamp, pos_obj, box_obj, vel_obj, acc_obj, rot_obj, ang_obj)
    return obj


def object_to_boxdetection(obj: ObjectState) -> BoxDetection:
    box_noisy = obj.box3d
    box_noisy.position[:2] += 0.10 * np.random.randn(2)
    det = BoxDetection(
        source_identifier="percep",
        box=box_noisy,
        reference=box_noisy.reference,
        obj_type=obj.obj_type,
        score=1.0,
    )
    return det


class AgentDetectionPublisher(Node):
    def __init__(self):
        super().__init__("agent_detection_publisher")
        self.publisher_dets = self.create_publisher(
            Detection3DArray, "detections_3d", 10
        )
        self.publisher_truths = self.create_publisher(ObjectStateArray, "truths", 10)
        self._timer = self.create_timer(0.1, self.publish_loop)
        self._t_reset = 100
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        tr = TransformStamped()
        tr.header.stamp = self.get_clock().now().to_msg()
        tr.header.frame_id = "world"
        tr.child_frame_id = "agent0"
        self.tf_static_broadcaster.sendTransform(tr)
        self._t0 = None

    def publish_loop(self):
        if self._t0 is None:
            self._t0 = Bridge.rostime_to_time(self.get_clock().now().to_msg())

        # populate header
        dt = Bridge.rostime_to_time(self.get_clock().now().to_msg()) - self._t0
        timestamp = Bridge.time_to_rostime(dt)
        header = Header()
        header.stamp = timestamp
        header.frame_id = "world"

        # create truth messages
        n_truths = 10
        timestamp = Bridge.rostime_to_time(header.stamp) % self._t_reset
        truths = [
            get_object_global(seed=i, timestamp=timestamp) for i in range(n_truths)
        ]
        msg_truths = ObjectStateBridge.avstack_to_objecstatearray(truths, header=header)
        self.publisher_truths.publish(msg_truths)

        # create detection messages
        np.random.seed(None)
        n_detections = n_truths
        dets = [object_to_boxdetection(obj) for obj in truths[:n_detections]]
        msg_dets = DetectionBridge.avstack_to_detectionarray(dets, header=header)
        self.publisher_dets.publish(msg_dets)


def main(args=None):
    rclpy.init(args=args)

    detandtruth = AgentDetectionPublisher()

    rclpy.spin(detandtruth)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detandtruth.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
