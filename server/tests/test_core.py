import unittest
import numpy as np
from server.core.parser import interpolate_slider_path
from server.core.portfolio import compute_map_portfolio_skills, compute_mechanical_portfolio

class TestCoreFunctions(unittest.TestCase):
    
    def test_interpolate_slider_path_basic(self):
        # Straight line interpolation
        control_points = [(0, 0), (100, 0)]
        # Halfway point
        pt = interpolate_slider_path(control_points, 50.0)
        self.assertAlmostEqual(pt[0], 50.0)
        self.assertAlmostEqual(pt[1], 0.0)

    def test_interpolate_slider_path_overshoot(self):
        # Overshooting the target length should return the last control point
        control_points = [(0, 0), (100, 0)]
        pt = interpolate_slider_path(control_points, 150.0)
        self.assertEqual(pt, (100, 0))

    def test_compute_map_portfolio_skills_clamping(self):
        # Extreme high features should clamp to 100
        extreme_high_feats = {
            'snap_aim_score': 100.0,
            'distance_p95': 500.0,
            'flow_aim_score': 100.0,
            'distance_p90': 500.0,
            'density_notes_per_sec': 20.0,
            'velocity_p95': 10.0,
            'streaming_score': 100.0,
            'time_delta_p5': 20.0,
            'duration_seconds': 600.0,
            'sliders_ratio': 1.0,
            'angle_std': 90.0,
            'finger_control_score': 100.0,
            'time_delta_std': 200.0,
            'circle_size': 7.0,
            'overall_difficulty': 10.0,
            'approach_rate': 10.0,
            'visual_density_score': 10.0,
            'slider_complexity_score': 10.0,
            'slider_multiplier': 3.0
        }
        skills = compute_map_portfolio_skills(extreme_high_feats)
        for skill_name, val in skills.items():
            self.assertTrue(5.0 <= val <= 100.0, f"{skill_name} is out of bounds: {val}")
            
    def test_compute_mechanical_portfolio_insufficient_data(self):
        # Less than 10 hit objects should return None
        hit_results = [{'hit': True, 'aim_distance': 1.0, 'timing_offset': 5.0, 'target_x': 256, 'target_y': 192, 'target_time': 1000}] * 5
        difficulty = {'CircleSize': 4.0, 'OverallDifficulty': 8.0, 'ApproachRate': 9.0}
        mech = compute_mechanical_portfolio(hit_results, difficulty)
        self.assertIsNone(mech)
