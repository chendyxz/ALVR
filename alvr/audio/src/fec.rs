use bytes::BytesMut;
use chrono::prelude::*;
use alvr_common::prelude::*;
use alvr_sockets::ReceivedPacket;
use reed_solomon_erasure::galois_8::ReedSolomon;

use crate::FrameHeader;

const NBVR_PACKET_TYPE_AUDIO_FRAME: u8 = 90;

const MAX_AUDIO_BUFFER_SIZE: usize = 16;

// fec数据最大20
const FEC_SHARDS_MAX: usize = 20;

// 数据块最大255
const DATA_SHARDS_MAX: usize = 255;

// fec百分比
const FEC_PERCENTAGE: usize = 50;

// 计算奇偶校验包数量
fn calculate_parity_shards(data_shards: usize, fec_percentage: usize) -> usize {
    let total_parity_shards = (data_shards * fec_percentage + 99) / 100;
    return total_parity_shards;
}

// Calculate how many packet is needed for make signal shard.
fn calculate_fec_shard_packets(len: usize, fec_percentage: usize) -> usize {
    // This reed solomon implementation accept only 255 shards.
    // Normally, we use NBVR_MAX_VIDEO_BUFFER_SIZE as block_size and single packet becomes single shard.
    // If we need more than maxDataShards packets, we need to combine multiple packet to make single shrad.
    // NOTE: Moonlight seems to use only 255 shards for video frame.
    let max_data_shards =
        ((FEC_SHARDS_MAX - 2) * 100 + 99 + fec_percentage) / (100 + fec_percentage);
    let min_block_size = (len + max_data_shards - 1) / max_data_shards;
    let shard_packets = (min_block_size + MAX_AUDIO_BUFFER_SIZE - 1) / FEC_SHARDS_MAX;
    assert!(
        max_data_shards + calculate_parity_shards(max_data_shards, fec_percentage)
            <= FEC_SHARDS_MAX
    );
    return shard_packets;
}

// 将数组转变为二维数组
fn construct_2d_array(original: Vec<u8>, m: usize, n: usize) -> Vec<Vec<u8>> {
    match original.len() as usize == m * n {
        false => Vec::new(),
        true => original
            .chunks(m as usize)
            .map(|v| v.to_vec())
            .collect::<Vec<Vec<u8>>>(),
    }
}

// fec 编码  -> Vec<(FrameHeader, Vec<u8>)>
pub fn encode(origin_data: Vec<u8>, findex: u32) -> Vec<(FrameHeader, Vec<u8>)> {
    // 原数据长度
    let length = origin_data.len();
    // fec包数量
    let shards_packets = calculate_fec_shard_packets(length, FEC_PERCENTAGE);
    // 每个数组长度
    let block_size = shards_packets * MAX_AUDIO_BUFFER_SIZE;
    // 媒体数组数量
    let data_shards = (length + block_size - 1) / block_size;
    // 总的奇偶校验包数量
    let total_parity_shards = calculate_parity_shards(data_shards, FEC_PERCENTAGE);
    // 总的数据数量（媒体包+冗余包）
    let total_shards = data_shards + total_parity_shards;
    // 总的数据包数要小于等于255
    assert!(total_shards <= DATA_SHARDS_MAX);

    // info!("reed_solomon_new. dataShards={} totalParityShards={} totalShards={} blockSize={} shardPackets={}"
    //      , data_shards, total_parity_shards, total_shards, block_size, shards_packets);

    // 初始化，传入媒体包个数和总的数据包个数（媒体包+奇偶效验包）
    let rs = ReedSolomon::new(data_shards, total_parity_shards).unwrap();

    // 将媒体包数据转换为二维数组，子数组长度为block_size，长度为data_shards
    let mut master_copy = construct_2d_array(origin_data, block_size, data_shards);

    //info!("audio---> {:?}", master_copy);

    // 初始化冗余包数组（内容为0的二维数组），子数组长度为block_size，长度为total_parity_shards
    let mut fec_blocks = vec![vec![0; block_size]; total_parity_shards];

    // 在媒体包后追加冗余包
    master_copy.append(&mut fec_blocks);

    //println!("audio2---> {:?}", master_copy);

    // FEC编码（将冗余包内容填充）
    // Construct the parity shards
    rs.encode(&mut master_copy).unwrap();

    //info!("audio encode --> {:?}", master_copy);

    //  media_type: u8,         // NBVR_PACKET_TYPE_AUDIO_FRAME
    //  packet_counter: usize,  // 总包数
    //  frame_index: usize,     // 帧索引
    //  sent_time: i64,         // 发送时间
    //  frame_byte_size: usize, // 帧大小
    //  packet_size: usize,     // 包大小
    //  packet_index: usize,    // 包索引
    //  fec_counter: usize,     // fec数据包个数
    //  fec_percentage: usize,  // fec百分比
    let mut out_data: Vec<(FrameHeader, Vec<u8>)> = Vec::with_capacity(master_copy.len());
    // let mut out_data: Vec<(FrameHeader, Vec<u8>)> = Vec::new();
    for (index, value) in master_copy.iter().enumerate() {
        let header = FrameHeader {
            media_type: NBVR_PACKET_TYPE_AUDIO_FRAME,
            packet_counter: rs.total_shard_count(),
            frame_index: findex,
            sent_time: Local::now().timestamp_millis(),
            frame_byte_size: value.len() * rs.total_shard_count(),
            packet_size: value.len(),
            packet_index: index,
            fec_counter: rs.parity_shard_count(),
            fec_percentage: FEC_PERCENTAGE,
            fec_index: rs.data_shard_count() + index,
            first_packet: if index == 0 { 1 } else { 0 },
        };
        //out_data[index] = (header, value.to_vec());
        out_data.push((header, value.to_vec()));
    }
    //info!("nane: out ==> {:?}", out_data);
    out_data
}

// fec解码
pub fn decode(
    mut origin_data: Vec<Option<Vec<u8>>>,
    data_shards: usize,
    total_parity_shards: usize,
) -> Vec<u8> {
    //info!("nane:decode start");
    //info!("nane: origin data => {}",origin_data.len());

    // 初始化，传入媒体包个数和总的数据包个数（媒体包+奇偶效验包）
    let rs = ReedSolomon::new(data_shards, total_parity_shards).unwrap();

    rs.reconstruct(&mut origin_data).unwrap();

    // info!(
    //     "nane: decode data_shards:{}, total_parity_shards:{}, test_data:{}",
    //     rs.data_shard_count(),
    //     rs.parity_shard_count(),
    //     origin_data.len()
    // );
    //info!("nane: decode success");

    // todo 记得修改数据源
    let mut result: Vec<_> = origin_data
        .into_iter()
        .filter_map(|x| x)
        .map(|s| s)
        .collect();

    // info!("nane: successs recovery data ==> {:?}", result);

    // 恢复成功返回数组
    match rs.verify(&result) {
        Ok(_val) => {
            //info!("nane: success");
            // 如果恢复成功,删除冗余包数据(此步操作不能放在match前, rs判断是否恢复成功需要使用原数据判断)
            result.drain(rs.data_shard_count()..);
            let ret = result.concat();
            //info!("nane: recovery ==> {:?}", ret);
            ret
        }
        Err(e) => {
            info!("nane: failed: {e}");
            // todo 恢复失败时,出去冗余包,返回媒体包,因包含了none类型,需要额外处理,暂未实现
            result.drain(rs.data_shard_count()..);
            result.concat()
        }
    }
}

pub fn reconstruct_data(
    header: FrameHeader,
    packets: Vec<ReceivedPacket<FrameHeader>>,
) -> BytesMut {
    // info!("nane: reconstruct start");
    // info!(
    //     "nane: total_packet: {}, recv_packet: {}",
    //     header.packet_counter,
    //     packets.len()
    // );

    // 丢包数大于fec数量,无法恢复则不使用fec, 反之则使用fec
    // 总包数(header.packet_counter) - 收到的包数(packets.len()) < fec冗余包数(header.fec_counter) 为true, 可以使用fec
    let enable_fec = header.packet_counter - packets.len() < header.fec_counter;

    let mut data: Vec<Option<Vec<u8>>> = vec![None; header.packet_counter];
    for (index, ele) in packets.into_iter().enumerate() {
        let samples = ele.buffer.freeze().to_vec();
        if index <= ele.header.packet_index {
            data[ele.header.packet_index] = Some(samples);
        } else {
            error!("index beyond");
        }
    }

    //info!("nane: reconstruct => {:?}", data);

    if data.contains(&None) {
        //info!("nane: audio packet loss");
        // 有丢包
        if enable_fec {
            // 丢包数小于fec包数量, 可以恢复
            let decode_vec = decode(
                data,
                header.packet_counter - header.fec_counter,
                header.fec_counter,
            );
            BytesMut::from(&decode_vec[..])
        } else {
            // 丢包数大于fec包数量, 无法恢复
            BytesMut::from(
                &vec![0; (header.packet_counter - header.fec_counter) * header.packet_size][..],
            )
        }
    } else {
        //info!("nane: audio no loss");
        // 无丢包
        let mut raw_data: Vec<_> = data.into_iter().filter_map(|s| s).map(|x| x).collect();
        // 删除冗余数据,只保留媒体数据
        raw_data.drain(header.packet_counter - header.fec_counter..);
        // 将二维数组转为一个数组([[0,0,0],[0,0,0],[0,0,0]] ==> [0,0,0,0,0,0,0,0,0])
        let res_vec = raw_data.concat();
        // 将数组转为byte
        BytesMut::from(&res_vec[..])
    }

    // let res: Vec<u8> = vec![1; 3];
    // BytesMut::from(&res[..])
}
