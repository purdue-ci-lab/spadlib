% Original code written by Pavlos (I think)

clear; close all; clc;

foldername = 'C:\Users\pconstas\Desktop\aqua_base\usrsrc\piccolo_raw_v2\laser_2MHz_0.02\';

file_range = 1:100;

if isempty(gcp('nocreate'))
    parpool('local', 12);
end

parfor idx = 1:numel(file_range)
    file_cnt = file_range(idx);

    try
        process_piccolo_file(foldername, file_cnt);
    catch ME
        warning("Failed file %d: %s", file_cnt-1, ME.message);
    end
end


function process_piccolo_file(foldername, file_cnt)

    filename = sprintf('raw_top_%d.bin', file_cnt-1);
    inpath = fullfile(foldername, filename);

    fid = fopen(inpath, 'rb');
    if fid < 0
        warning("Could not open %s", inpath);
        return;
    end

    c = fread(fid, 'uint32');
    fclose(fid);

    if numel(c) < 11
        warning("File %s too short", inpath);
        return;
    end

    cc = c;
    c2 = c;

    data_counts = zeros(1,4);
    data_counts(1) = bitand(c2(4),  2^22 - 1);
    data_counts(2) = data_counts(1) + bitand(c2(6),  2^22 - 1);
    data_counts(3) = data_counts(2) + bitand(c2(8),  2^22 - 1);

    total_photons = bitand(c2(4),  2^22 - 1) + ...
                    bitand(c2(6),  2^22 - 1) + ...
                    bitand(c2(8),  2^22 - 1) + ...
                    bitand(c2(10), 2^22 - 1);

    expected_photons = bitand(c2(2), 2^22 - 1);

    if total_photons ~= expected_photons
        warning("Skipping file %d: inconsistent photon number", file_cnt-1);
        return;
    end

    c2 = c2(11:length(cc));

    c2(end+1) = 2^32 - 1;
    c2(end+1) = 2^32 - 1;

    c2p = c2;
    c2p_ysize = 8 * floor(length(c2p) / 8);
    c2s = reshape(c2p(1:c2p_ysize), [8, c2p_ysize/8]);

    c2r = c2s([1 2 3 4 5 6 7 8], :);
    c2b = reshape(c2r, [c2p_ysize, 1]);
    c2 = c2b;

    c3 = reshape(c2, 2, []);
    tdcs = bitshift(bitand(c3(1,:), 6), -1);
    c3(1,:) = bitand(c3(1,:), 2^32 - 8);

    c4 = squeeze(c3(2,:));
    c = c4;

    d = zeros(4096, 32, 32, 'uint32');

    c5 = zeros(1, length(c4));
    c5col = zeros(length(c4), 32);

    event_fpga = 0;
    coarse_cnt_ovfs = zeros(1,4);
    event_fpga_col = zeros(1,32);

    ch = 1;
    prev_event = 0;
    prev_event_index = 0;

    nmax = length(c) - 1;

    rows = zeros(1, nmax);
    cols = zeros(1, nmax);
    tdc_fine = zeros(1, nmax);
    tdc_coarse = zeros(1, nmax);
    tdc_ids = zeros(1, nmax);

    readout_stamp = zeros(1, max(262144, nmax));

    for i = 1:nmax
        ci = c(i);

        if floor(ci / 2147483648) == 0

            fine = mod(ci, 4096) + 1;
            row  = bitand(floor(ci / 4096), 31) + 1;
            col  = bitand(floor(ci / 4096 / 32), 31) + 1;

            d(fine, row, col) = d(fine, row, col) + 1;

            event_fpga = event_fpga + 1;
            event_fpga_col(col) = event_fpga_col(col) + 1;

            rows(event_fpga) = row;
            cols(event_fpga) = col;
            tdc_fine(event_fpga) = fine;
            tdc_ids(event_fpga) = tdcs(i);

            if ((c3(1,i) + 100 < prev_event) && (i > prev_event_index + 16))
                coarse_cnt_ovfs(ch) = coarse_cnt_ovfs(ch) + 1;
                prev_event_index = i;
            end

            c5col(event_fpga_col(col), col) = c3(1,i) + coarse_cnt_ovfs(ch) * 2^16;

            if i > prev_event_index + 8
                coarse_val = c3(1,i) + coarse_cnt_ovfs(ch) * 2^16;
            else
                if mod(c3(1,i), 2^16 - 1) > 2^15
                    coarse_val = c3(1,i) + (coarse_cnt_ovfs(ch) - 1) * 2^16;
                else
                    coarse_val = c3(1,i) + coarse_cnt_ovfs(ch) * 2^16;
                end
            end

            c5(event_fpga) = coarse_val;
            tdc_coarse(event_fpga) = coarse_val;

            prev_event = c3(1,i);

            if ch <= 3 && i == data_counts(ch)
                ch = ch + 1;
                prev_event = 0;
                prev_event_index = 0;
            end
        end

        readout_stamp(i) = floor(c3(2,i) / 2^24);
    end

    rows = rows(1:event_fpga);
    cols = cols(1:event_fpga);
    tdc_fine = tdc_fine(1:event_fpga);
    tdc_coarse = tdc_coarse(1:event_fpga);

    timestampsCounter1 = cell(32, 32);
    timestampsCounter2 = cell(32, 32);
    events_image = zeros(32, 32);
    timestampsTdcId = cell(32, 32);
    for row = 1:32
        for col = 1:32
            pixel_indices = (rows == row) & (cols == col);
            timestampsTdcId{row, col} = tdc_ids(pixel_indices);
            timestampsCounter1{row, col} = tdc_coarse(pixel_indices);
            timestampsCounter2{row, col} = tdc_fine(pixel_indices);
            events_image(row, col) = numel(timestampsCounter2{row, col});
        end
    end

    outname = sprintf('timestamps_%d.mat', file_cnt-1);
    outpath = fullfile(foldername, outname);

    save(outpath, 'timestampsCounter1', 'timestampsCounter2','timestampsTdcId',  'events_image');
end
