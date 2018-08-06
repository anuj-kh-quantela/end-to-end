from car_analytics_without_roi import CarAnalytics
vo = CarAnalytics('/media/anuj/Work-HDD/WORK/CLOUD-DRIVE/Google-Drive/Computer-Vision/Sample-Videos/Car-Analytics/red_light_sim_1.mp4', 'vijaywada', 'benzcircle')
vo.setup_counting_cars()
# vo.new_setup_high_speed_detection()
# vo.new_setup_wrong_direction_detection()
# vo.new_setup_traffic_signal_violation_detection()
vo.run()