import re
import numpy as np


class BeamDescription:
    def __init__(self, beam_profile, start_points, end_points):
        # init beam profile
        self.shape = beam_profile.shape
        self.divergent = beam_profile.divergent
        self.parameter_start = beam_profile.parameter_start
        self.parameter_end = beam_profile.parameter_end
        # init actual beam
        self.start_points = start_points
        self.end_points = end_points

    @staticmethod
    def get_description_from_files(beam_profile_file, beam_file):
        # read start and end points
        read_lines = 0
        error_limit = 3
        edge_points = None
        while True:
            try:
                float_list_from_a_line = get_float_list_from_a_line(beam_file)
                if edge_points is None:
                    edge_points = np.array(float_list_from_a_line)
                else:
                    edge_points = np.array([edge_points] + [float_list_from_a_line]) * 1e-2
                read_lines = read_lines + 1
            except Exception as e:
                error_limit = error_limit - 1
                if error_limit == 0:
                    print "Something is wrong"
                    break
                continue
            if read_lines == 2:
                break

        # create BeamProfile
        beam_profile = BeamProfile(beam_profile_file)

        return BeamDescription(beam_profile, edge_points[0], edge_points[1])


class BeamProfile:
    def __init__(self, beam_profile_file):
        try:
            beam_profile_file.readline()
            self.shape = int(re.search(r'\d+', beam_profile_file.readline()).group())
            self.divergent = int(re.search(r'\d+', beam_profile_file.readline()).group())
            self.parameter_start = get_float_list_from_a_line(beam_profile_file)/100
            self.parameter_end = get_float_list_from_a_line(beam_profile_file)/100
        except AttributeError as e:
            print "File not good"

        pass


def get_float_list_from_a_line(beam_profile_file):
    read_line = beam_profile_file.readline()
    return np.array(read_line.strip().split()).astype(np.float)

