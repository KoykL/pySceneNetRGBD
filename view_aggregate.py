import camera_pose_and_intrinsics_example as camera
import numpy as np
from PIL import Image
import os
import argparse
import scenenet_pb2 as sn
import tensorflow as tf
import time
WIDTH = 320
HEIGHT = 240
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "..", ".."))))
import utils.pointutils
def unproj(trajectories, opts):
    #https://stackoverflow.com/questions/31265245/extracting-3d-coordinates-given-2d-image-points-depth-map-and-camera-calibratio
    intrinsic_matrix = camera.camera_intrinsic_transform()
    unproj_mat = np.linalg.inv(intrinsic_matrix[:3, :3])
    data_prefix = opts.data_path
    for traj  in trajectories.trajectories:
        start = time.time()
        for idx,view in enumerate(traj.views):
            points_path = os.path.join(data_prefix, traj.render_path, "points")
            points_file = os.path.join(points_path, "{}.npy".format(view.frame_num))
            if opts.skip_exists and os.path.exists(points_file):
                print("skipping {}".format(points_file))
                continue
            
            #print('world_to_camera_matrix: \n', world_to_camera_matrix) 
            #coords_aug = np.insert(coords, 1, axis=2)
            
            #coords = np.mgrid[0:HEIGHT, 0:WIDTH].transpose(1, 2, 0).reshape(-1, 2)
            
            depth_file = os.path.join(data_prefix, traj.render_path, "depth", "{}.png".format(view.frame_num))
            
            depth = np.array(Image.open(depth_file), dtype=np.float) / 1000
            depth = depth.reshape(-1)
            focal = np.array([intrinsic_matrix[1, 1], intrinsic_matrix[0, 0]])
            center = np.array([intrinsic_matrix[1, 2], intrinsic_matrix[0, 2]])

            # Get camera pose
            ground_truth_pose = camera.interpolate_poses(view.shutter_open,view.shutter_close,0.5)
            camera_to_world_matrix = camera.camera_to_world_with_pose(ground_truth_pose)

            xs_tf = tf.range(0, WIDTH)
            ys_tf = tf.range(0, HEIGHT)
            xs_tf, ys_tf = tf.meshgrid(xs_tf, ys_tf)
            coords_tf = tf.reshape(tf.transpose(tf.stack([ys_tf, xs_tf]), [1, 2, 0]), [-1, 2])
            coords_tf = tf.cast(coords_tf, dtype=tf.float32)
            center_tf = tf.convert_to_tensor(center, dtype=tf.float32)
            focal_tf = tf.convert_to_tensor(focal, dtype=tf.float32)
            depth_tf = tf.convert_to_tensor(depth, dtype=tf.float32)
            
            camera_to_world_matrix_tf = tf.convert_to_tensor(camera_to_world_matrix, dtype=tf.float32)
            coords_3d_tf_yx = (coords_tf - center_tf) * tf.expand_dims(depth_tf, axis=-1) / focal_tf
            coords_3d_tf = tf.concat([tf.reverse(coords_3d_tf_yx, axis=[1]), tf.expand_dims(depth_tf, axis=-1)], axis=1)
            coords_3d_tf_aug = tf.concat([coords_3d_tf, tf.constant(1, dtype=tf.float32, shape=[coords_3d_tf.shape[0], 1])], axis=-1)
            coords_3d_tf_aug_world = tf.transpose(tf.matmul(camera_to_world_matrix_tf, coords_3d_tf_aug, transpose_b=True))
            coords_3d_tf_world = coords_3d_tf_aug_world[:, :3] / coords_3d_tf_aug_world[:, 3:4]
            #coords_3d = (coords - center) * np.expand_dims(depth, axis=-1) / focal
            os.makedirs(points_path, exist_ok=True)
            #np.save(points_file, coords_3d)
            np.save(points_file, coords_3d_tf_world.numpy())
            if opts.vis_points:
                points_vis_path = os.path.join(data_prefix, traj.render_path, "points_vis")
                os.makedirs(points_vis_path, exist_ok=True)
                points_vis_file = os.path.join(points_vis_path, "{}.obj".format(view.frame_num))
                with open(points_vis_file, "wt") as f:
                    utils.pointutils.dump_obj(coords_3d_tf_world.numpy(), f)
                    pass
                pass
            pass
        end = time.time()
        print("loop time: ", end-start)
        return
        pass
    pass

def has_overlap_view(traj, opts):
    data_prefix = opts.data_path
    
    for idx1,view1 in enumerate(traj.views):
        for idx2,view2 in enumerate(traj.views):
            points_file_1 = os.path.join(data_prefix, traj.render_path, "points", "{}.npy".format(view1.frame_num))
            points_file_2 = os.path.join(data_prefix, traj.render_path, "points", "{}.npy".format(view2.frame_num))
            points_1 = np.load(points_file_1)
            points_2 = np.load(points_file_2)
            points_1_tf = tf.convert_to_tensor(points_1, dtype=tf.float32)
            points_2_tf = tf.convert_to_tensor(points_2, dtype=tf.float32)
            points_1_ex = tf.expand_dims(points_1_tf, axis=1)
            points_2_ex = tf.expand_dims(points_2_tf, axis=0)
            distances = tf.norm(points_1_ex - points_2_ex, ord=2, axis=-1)
            min_dist = tf.math.reduce_min(distances)
            print(min_dist)
            pass
        pass
    pass

def main():
    trajectories = sn.Trajectories()
    parser = argparse.ArgumentParser()
    parser.add_argument("protobuf_path", type=str, help="path of the protobuf file")
    parser.add_argument("data_path", type=str, help="path of the data")
    parser.add_argument("--vis_points", default=False, action="store_true", help="path of the data")
    parser.add_argument("--skip_exists", default=True, action="store_false", help="skip existing files")
    args = parser.parse_args()
    protobuf_path = args.protobuf_path
    data_path = args.data_path
    try:
        with open(protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(data_root_path))
        print('Please ensure you have copied the pb file to the data directory')
        pass
    unproj(trajectories, args)
    # start = time.time()
    # has_overlap_view(trajectories.trajectories[0], args)
    # end = time.time()
    # print("overlap time: {}".format(end - start))
    pass
if __name__ == "__main__":
    main()
