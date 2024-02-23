import copy
import os.path
import cv2
import json
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

DEBUG = True
RED = (255, 0, 0)
BLUE = (0, 0, 255)
resize = (700, 500)

def create_directory(directory_path):
    """
    Check if a directory exists, if not, create it.
    :param directory_path: The path of the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        # print(f"Folder '{directory_path}' created successfully")
    else:
        # print(f"Folder '{directory_path}' already exists")
        pass



def get_object_tracking() -> list:
    object_tracking_json = os.path.join("part_1_object_tracking.json")
    with open(object_tracking_json, 'r') as f:
        object_tracking_data = json.load(f)

    object_tracking_obj_data = object_tracking_data['obj']
    return object_tracking_obj_data


def get_video() -> cv2.VideoCapture:
    video = os.path.join("commonwealth.mp4")

    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # if count != len(object_tracking_obj_data):
    #     raise Exception("frame count error")

    print(f"{video} w: {width} h: {height} fps: {fps} nums: {count}")
    return cap


def create_output(cap: cv2.VideoCapture, name="output.mp4") -> cv2.VideoCapture:
    fourcc =  cv2.VideoWriter_fourcc(*'mp4v')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    width = resize[0]
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = resize[1]
    fps = cap.get(cv2.CAP_PROP_FPS)
    output = cv2.VideoWriter(name, fourcc, fps, (int(width), int(height)))
    return output


def con_list(input: list) -> list:
    indexs = []
    index = [0, 0]
    continuous_sequence = False
    start_index = None

    for i, num in enumerate(input):
        if num != [-1, -1]:
            if not continuous_sequence:
                start_index = i
                continuous_sequence = True
        else:
            if continuous_sequence:
                index[0] = start_index
                index[1] = i - 1
                indexs.append(index)
                continuous_sequence = False

    if continuous_sequence:
        index[0] = start_index
        index[1] = len(input) - 1
        indexs.append(index)

    return indexs

def smooth_data2(draw_tracking: list) -> list:
    copy_tracking = copy.deepcopy(draw_tracking)
    mask = [[1, 1]if point == [-1, -1] else [0, 0] for point in draw_tracking]
    raw = np.asarray(copy_tracking)
    measurements = np.ma.masked_array(copy_tracking,
                                      mask=mask)
    observed_data = measurements.copy()
    # initial_state_mean = [0, 0]
    initial_point = [0, 0]
    for point in measurements:
        if np.array_equal(point, [-1, -1]):
            continue
        initial_point = point
        break
    initial_state_mean = [initial_point[0], initial_point[1]]
    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]
    kf = KalmanFilter(initial_state_mean=initial_state_mean,
                      n_dim_obs=2,
                      observation_covariance=100 * np.eye(2))  # 增大观测噪声
                      # transition_matrices=transition_matrix,
                      # observation_matrices=observation_matrix)

    smoothed_state_means, smoothed_state_covariances = kf.smooth(observed_data)
    frame_index = [i for i in range(len(copy_tracking))]
    if DEBUG:
        plt.plot(frame_index, raw[:, 0], 'bo', label="x")
        plt.plot(frame_index, raw[:, 1], 'ro', label="y")
        plt.plot(frame_index, smoothed_state_means[:, 0], 'b--', label="s-x")
        plt.plot(frame_index, smoothed_state_means[:, 1], 'r--', label="s-y")
        plt.legend()
        create_directory("debug")
        plt.savefig(os.path.join("debug", "smoothed2.png"))
        # plt.show()
    return smoothed_state_means.round()

def smooth_draw_tracking(draw_tracking):
    copy_tracking = copy.deepcopy(draw_tracking)
    indexs = con_list(draw_tracking)
    for start, end in indexs:
        input = copy_tracking[start: end + 1]
        measurements = np.asarray(input)

        initial_state_mean = [measurements[0, 0],
                              0,
                              measurements[0, 1],
                              0]

        transition_matrix = [[1, 1, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, 0, 1]]

        observation_matrix = [[1, 0, 0, 0],
                              [0, 0, 1, 0]]

        kf1 = KalmanFilter(transition_matrices=transition_matrix,
                           observation_matrices=observation_matrix,
                           initial_state_mean=initial_state_mean)

        kf1 = kf1.em(measurements, n_iter=5)
        (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

        copy_tracking[start: end + 1] = [[x, y] for x, y in zip(smoothed_state_means[:, 0], smoothed_state_means[:, 2])]

    times = range(len(draw_tracking))
    raw = np.asarray(draw_tracking)
    res = np.asarray(copy_tracking)
    plt.plot(times, raw[:, 0], 'bo',
             times, raw[:, 1], 'ro',
             times, res[:, 0], 'b--',
             times, res[:, 1], 'r--', )
    # plt.show()
    1


def draw_tracking(cap: cv2.VideoCapture, output: cv2.VideoWriter, smooth_output: cv2.VideoWriter, draw_tracking: list,
                  smooth_draw_tracking: list):
    i = 0
    coordinates = []
    smooth_coordinates = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # print(f"-->{i}")

        point = draw_tracking[i]
        if point != [-1, -1]:
            # print(point)
            out_frame = copy.deepcopy(frame)
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            out_frame = cv2.resize(out_frame, resize)

            coordinate = tuple(point)
            coordinates.append(coordinate)

            for j in range(1, len(coordinates)):
                cv2.line(out_frame, (int(coordinates[j - 1][0]), int(coordinates[j - 1][1])),
                         (int(coordinates[j][0]), int(coordinates[j][1])), BLUE, 2)
            w, h = 4, 4
            cv2.rectangle(out_frame, (int(coordinate[0] - w / 2), int(coordinate[1] - h / 2)),
                          (int(coordinate[0] + w / 2), int(coordinate[1] + h / 2)), RED, 2)

            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            if DEBUG:
                cv2.imwrite(f'debug/frame_{i}.jpg', out_frame)
            output.write(out_frame)

        smooth_point = smooth_draw_tracking[i]
        if not np.array_equal(smooth_point ,[-1, -1]):
            # print(smooth_point)

            smooth_out_frame = copy.deepcopy(frame)
            smooth_out_frame = cv2.cvtColor(smooth_out_frame, cv2.COLOR_BGR2RGB)
            smooth_out_frame = cv2.resize(smooth_out_frame, resize)

            coordinate = tuple(smooth_point)
            smooth_coordinates.append(coordinate)

            for j in range(1, len(smooth_coordinates)):
                cv2.line(smooth_out_frame, (int(smooth_coordinates[j - 1][0]), int(smooth_coordinates[j - 1][1])),
                         (int(smooth_coordinates[j][0]), int(smooth_coordinates[j][1])), BLUE, 2)
            w, h = 4, 4
            cv2.rectangle(smooth_out_frame, (int(coordinate[0] - w / 2), int(coordinate[1] - h / 2)),
                          (int(coordinate[0] + w / 2), int(coordinate[1] + h / 2)), RED, 2)

            smooth_out_frame = cv2.cvtColor(smooth_out_frame, cv2.COLOR_RGB2BGR)
            if DEBUG:
                cv2.imwrite(f'debug/smooth_frame_{i}.jpg', smooth_out_frame)
            smooth_output.write(smooth_out_frame)
        # cv2.imshow('Tracked Path', out_frame)
        # press 'q' exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

        i += 1


def run():
    object_tracking_obj_data = get_object_tracking()
    smooth_object_tracking_data = smooth_data2(object_tracking_obj_data)
    cap = get_video()
    output = create_output(cap, name="output.mp4")
    smooth_output = create_output(cap, name="smooth_output.mp4")
    draw_tracking(cap, output, smooth_output, object_tracking_obj_data, smooth_object_tracking_data)
    cap.release()
    output.release()
    cv2.destroyAllWindows()


def smooth_data(object_tracking_obj_data: list) -> list:
    # print the smooth
    object_tracking_obj_data = get_object_tracking()
    has_value = np.asarray([[x, y] for x, y in object_tracking_obj_data if x != -1 and y != -1])
    # has_value = np.ma.masked_array(object_tracking_obj_data,
    #                    mask=[[1, 1] if p == [-1, -1] else [0, 0] for p in object_tracking_obj_data])
    frame_index = [i for i in range(len(object_tracking_obj_data)) if object_tracking_obj_data[i] != [-1, -1]]

    measurements = np.asarray(has_value)

    initial_state_mean = [measurements[0, 0],
                          0,
                          measurements[0, 1],
                          0]

    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

    kf1 = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=initial_state_mean)

    kf1 = kf1.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

    if DEBUG:
        plt.plot(frame_index, has_value[:, 0], 'bo', label="x")
        plt.plot(frame_index, has_value[:, 1], 'ro', label="y")
        plt.plot(frame_index, smoothed_state_means[:, 0], 'b--', label="s-x")
        plt.plot(frame_index, smoothed_state_means[:, 2], 'r--', label="s-y")
        plt.legend()
        create_directory("debug")
        plt.savefig(os.path.join("debug", "smoothed.png"))
        # plt.show()

    result = copy.deepcopy(object_tracking_obj_data)
    j = 0
    for i in frame_index:
        result[i] = [int(smoothed_state_means[j, 0]), int(smoothed_state_means[j, 2])]
        j += 1

    data = {"obj": [x for x in result if x != [-1, -1]]}
    file_path = "smooth_tracking.json"

    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
    return result


if __name__ == '__main__':
    run()
    print("success")
