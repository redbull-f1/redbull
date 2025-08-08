#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
import argparse
import shutil
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

"""
ROS2 bag 파일 병합 도구
- ego bag: /scan, /car_state/odom, /tf, /tf_static
- opp bag: /opp/car_state/odom
- 겹치는 시간 구간만 병합

# 기본 사용 (요청하신 명령어)
python3 /home/harry/ros2_ws/src/redbull/merge_bags.py --ego-bag ego.db3 --opp-bag opp.db3

# 출력 파일명 지정
python3 /home/harry/ros2_ws/src/redbull/merge_bags.py --ego-bag ego.db3 --opp-bag opp.db3 --output my_merged.db3
"""

class BagMerger(Node):
    """
    두 개의 ROS2 bag 파일을 시간 동기화하여 병합
    """
    
    def __init__(self, ego_bag_path, opp_bag_path, output_bag_path):
        super().__init__('bag_merger')
        
        self.ego_bag_path = ego_bag_path
        self.opp_bag_path = opp_bag_path
        self.output_bag_path = output_bag_path
        
        # 처리할 토픽 정의
        self.ego_topics = ["/scan", "/car_state/odom", "/tf", "/tf_static"]
        self.opp_topics = ["/opp/car_state/odom"]
        
    def read_bag_messages(self, bag_path, target_topics):
        """
        bag 파일에서 지정된 토픽의 메시지들을 읽어옴
        """
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        
        reader = SequentialReader()
        messages = []
        
        try:
            reader.open(storage_options, converter_options)
            
            topic_types = reader.get_all_topics_and_types()
            
            # 메시지 타입 매핑
            type_map = {}
            for topic_metadata in topic_types:
                type_map[topic_metadata.name] = topic_metadata.type
            
            self.get_logger().info(f'Bag 읽기 시작: {bag_path}')
            
            message_count = 0
            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()
                
                # 지정된 토픽만 처리
                if topic in target_topics and topic in type_map:
                    # 타임스탬프를 초 단위로 변환
                    timestamp_sec = timestamp / 1e9
                    
                    # 메시지 저장
                    messages.append({
                        'topic': topic,
                        'data': data,
                        'timestamp': timestamp,
                        'timestamp_sec': timestamp_sec,
                        'type': type_map[topic]
                    })
                    
                    message_count += 1
                    if message_count % 1000 == 0:
                        self.get_logger().info(f'읽은 메시지: {message_count}개')
            
            self.get_logger().info(f'Bag 읽기 완료: {message_count}개 메시지')
            return messages
            
        except Exception as e:
            self.get_logger().error(f'Bag 파일 읽기 중 오류 발생: {e}')
            return []
        finally:
            del reader
    
    def merge_bags(self):
        """
        두 bag 파일을 읽고 겹치는 시간 구간만 병합
        """
        # 1. 각 bag 파일에서 메시지 읽기
        self.get_logger().info('=== Bag 파일 읽기 시작 ===')
        
        ego_messages = self.read_bag_messages(self.ego_bag_path, self.ego_topics)
        opp_messages = self.read_bag_messages(self.opp_bag_path, self.opp_topics)
        
        if not ego_messages and not opp_messages:
            self.get_logger().error('읽을 수 있는 메시지가 없습니다!')
            return False
        
        # 2. 겹치는 시간 구간 계산
        ego_start = min(msg['timestamp_sec'] for msg in ego_messages) if ego_messages else float('inf')
        ego_end = max(msg['timestamp_sec'] for msg in ego_messages) if ego_messages else float('-inf')
        opp_start = min(msg['timestamp_sec'] for msg in opp_messages) if opp_messages else float('inf')
        opp_end = max(msg['timestamp_sec'] for msg in opp_messages) if opp_messages else float('-inf')
        
        # 겹치는 시간 구간: 더 늦게 시작하고 더 빨리 끝나는 시간
        sync_start_time = max(ego_start, opp_start)
        sync_end_time = min(ego_end, opp_end)
        
        self.get_logger().info(f'Ego 시간 범위: {ego_start:.3f}s ~ {ego_end:.3f}s')
        self.get_logger().info(f'Opp 시간 범위: {opp_start:.3f}s ~ {opp_end:.3f}s')
        self.get_logger().info(f'병합 시간 범위: {sync_start_time:.3f}s ~ {sync_end_time:.3f}s')
        
        # 겹치는 시간이 있는지 확인
        if sync_start_time >= sync_end_time:
            self.get_logger().error('두 파일의 시간이 겹치지 않습니다!')
            return False
        
        # 3. 겹치는 시간 범위 내의 메시지만 필터링
        ego_filtered = [msg for msg in ego_messages 
                       if sync_start_time <= msg['timestamp_sec'] <= sync_end_time]
        opp_filtered = [msg for msg in opp_messages 
                       if sync_start_time <= msg['timestamp_sec'] <= sync_end_time]
        
        self.get_logger().info(f'필터링 후 메시지 수 - Ego: {len(ego_filtered)}, Opp: {len(opp_filtered)}')
        
        if not ego_filtered and not opp_filtered:
            self.get_logger().error('겹치는 시간 범위 내의 메시지가 없습니다!')
            return False
        
        # 4. 모든 메시지를 시간순으로 병합
        all_messages = ego_filtered + opp_filtered
        all_messages.sort(key=lambda x: x['timestamp'])
        
        self.get_logger().info(f'총 {len(all_messages)}개 메시지를 시간순으로 정렬 완료')
        
        # 5. 병합된 bag 파일 생성
        return self.write_merged_bag(all_messages)
    
    def write_merged_bag(self, sorted_messages):
        """
        정렬된 메시지들을 새로운 bag 파일로 저장
        """
        if not sorted_messages:
            self.get_logger().error('저장할 메시지가 없습니다!')
            return False
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_bag_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        storage_options = StorageOptions(uri=self.output_bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        
        writer = SequentialWriter()
        
        try:
            writer.open(storage_options, converter_options)
            
            # 토픽 정보 수집
            topic_info_map = {}
            for msg in sorted_messages:
                topic = msg['topic']
                msg_type = msg['type']
                if topic not in topic_info_map:
                    topic_info_map[topic] = msg_type
                    # 토픽 생성 - TopicMetadata 객체 사용
                    topic_metadata = TopicMetadata(
                        name=topic,
                        type=msg_type,
                        serialization_format='cdr'
                    )
                    writer.create_topic(topic_metadata)
            
            self.get_logger().info(f'생성된 토픽들: {list(topic_info_map.keys())}')
            
            # 메시지 작성
            written_count = 0
            for msg in sorted_messages:
                try:
                    writer.write(msg['topic'], msg['data'], msg['timestamp'])
                    written_count += 1
                    
                    if written_count % 1000 == 0:
                        self.get_logger().info(f'저장된 메시지: {written_count}개')
                        
                except Exception as e:
                    self.get_logger().warning(f'메시지 저장 중 오류: {e}')
                    continue
            
            self.get_logger().info(f'병합 완료: {written_count}개 메시지가 저장됨')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Bag 파일 저장 중 오류 발생: {e}')
            return False
        finally:
            del writer
    

def main():
    parser = argparse.ArgumentParser(description='ROS2 bag 파일을 병합')
    parser.add_argument('--ego-bag', type=str, required=True, help='Ego 차량의 bag 파일 경로 (db3)')
    parser.add_argument('--opp-bag', type=str, required=True, help='Opponent 차량의 bag 파일 경로 (db3)')
    parser.add_argument('--output', type=str, help='출력 bag 파일 경로')
    
    args = parser.parse_args()
    
    # 출력 경로 설정
    if args.output is None:
        ego_name = os.path.basename(args.ego_bag).split('.')[0]
        opp_name = os.path.basename(args.opp_bag).split('.')[0]
        args.output = f'/home/harry/Downloads/merged_{ego_name}_{opp_name}.db3'
    
    # bag 파일 존재 확인
    if not os.path.exists(args.ego_bag):
        print(f'Error: ego bag 파일을 찾을 수 없습니다: {args.ego_bag}')
        return
    
    if not os.path.exists(args.opp_bag):
        print(f'Error: opp bag 파일을 찾을 수 없습니다: {args.opp_bag}')
        return
    
    # ROS2 초기화
    rclpy.init()
    
    try:
        # Bag 병합기 생성
        merger = BagMerger(args.ego_bag, args.opp_bag, args.output)
        
        print(f'Bag 파일 병합 시작:')
        print(f'  Ego bag: {args.ego_bag}')
        print(f'  Opp bag: {args.opp_bag}')
        print(f'  Output: {args.output}')
        
        if merger.merge_bags():
            print(f'성공적으로 병합되었습니다: {args.output}')
        else:
            print('병합에 실패했습니다.')
    
    except Exception as e:
        print(f'오류 발생: {e}')
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
