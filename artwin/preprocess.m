%% Setup
this_file_path = mfilename('fullpath');
idx = strfind(this_file_path,filesep);
this_file_parent = this_file_path(1:idx(end));

cutouts_root = fullfile(this_file_parent, 'cutouts');
artwin_root = fullfile('/nfs','projects','artwin', 'SIE factory datasets');
output_root = fullfile('/nfs','projects','artwin', 'experiments', 'thesis', 'artwin_as_inloc');

debug = false;
as_colmap = true;

restoredefaultpath;
addpath(cutouts_root);
addpath(fullfile(cutouts_root, 'codes'));

%% Code
cameras = containers.Map('KeyType', 'int64', 'ValueType', 'any');
points3D = containers.Map('KeyType', 'int64', 'ValueType', 'any');

if ~as_colmap
    rng(42);
    train_val_split_ratio = 0.2;
    train_path = fullfile(output_root, 'train');
    val_path = fullfile(output_root, 'val');
end

f = 800/tan(30*pi/180);         % focal length so that FOV is 60 dg
u0 = 1600/2;                    % principal point u0
v0 = 1200/2;                    % principal point v0
img_size = [1600 1200];         % image size
K = [f 0 u0; 0 f v0; 0 0 1];    % the calibration matrix

% All panoamas are processed with one camera.
cameras(1) = struct(                ...
    'camera_id', 1,                 ...
    'model', 'SIMPLE_PINHOLE',      ...
    'width', img_size(1),           ...
    'height', img_size(2),          ...
    'params', [f u0 v0]             ...
);

pano_count = 0;
listing = dir(fullfile(artwin_root, 'proc'));
for folder = 1:length(listing)
    name = listing(folder).name;
    if name == "." || name == ".."
        continue
    end
    parent = listing(folder).folder;
    pts_file = fullfile(parent, name, strcat(name, '_txt.ply'));
    pano_file = fullfile(parent, name, 'pano', 'pano-poses.csv');

    if ~as_colmap
        pts = load_pts(pts_file);
    end

    [pano_images, pano_poses, pano_C, pano_q] = load_pano_poses(pano_file);

    images = containers.Map('KeyType', 'int64', 'ValueType', 'any');

    for pano_id = 1:length(pano_images)
        panorama = pano_images{pano_id};
        pano_img = imread(fullfile(parent, name, 'pano', panorama));
        R2 = q2r(pano_q(:, pano_id));
        C2 = pano_C(:, pano_id);
        [X,Y] = meshgrid(0:img_size(1) - 1, 0:img_size(2) - 1);
        X = X(:);
        Y = Y(:);

        % Degrees used because the combination is used in file names.
        z_rot_angles = 0:30:330;
        %
        % Rotating only in the horizontal plane now (90dg is added to x below).
        %
        x_rot_angles = -30:30:30;
        for zi = 1:length(z_rot_angles)
            % if ~(name ~= "2019-09-28_08.31.29" && pano_id > 39 && z_rot_angles(zi) < 180)
            %     continue
            % end

            for xi = 1:length(x_rot_angles)
                z_rot_angle = z_rot_angles(zi) * pi / 180;
                % Add pi / 2 to rotate the view to the horizontal plane.
                x_rot_angle = (x_rot_angles(xi) + 90) * pi / 180;
                z_rot_mat = ...
                    [cos(z_rot_angle) -sin(z_rot_angle) 0; ...
                     sin(z_rot_angle)  cos(z_rot_angle) 0; ...
                     0                 0                1];
                x_rot_mat = ...
                   [1 0                 0;                ...
                    0 cos(x_rot_angle) -sin(x_rot_angle); ...
                    0 sin(x_rot_angle)  cos(x_rot_angle)];
                R = z_rot_mat * x_rot_mat;

                [status, msg, msgID] = mkdir(output_root);
                if ~as_colmap
                    [status, msg, msgID] = mkdir(train_path);
                    [status, msg, msgID] = mkdir(val_path);

                    if rand < train_val_split_ratio
                        save_root_path = val_path;
                    else
                        save_root_path = train_path;
                    end
                else
                    save_root_path = fullfile(output_root, name);
                    [status, msg, msgID] = mkdir(fullfile(save_root_path, "sparse"));
                end

                stem = sprintf( ...
                    "%s_%s_x%d_z%d", ...
                    name, ...
                    extractBefore(panorama, "-pano.jpg"), ...
                    x_rot_angles(xi), ...
                    z_rot_angles(zi) ...
                );

                reference_img_path = fullfile(save_root_path, sprintf("%s_reference.png", stem));

                if exist(reference_img_path, 'file') ~= 2

                    proj = R * inv(K) * [X, Y, ones(length(X), 1)]';
                    proj = proj .* (ones(3, 1) * 1 ./ sqrt(sum(proj .^ 2)));
                    beta_proj = - asin(proj(3, :));
                    cos_beta_proj = real(cos(beta_proj));
                    alpha_proj = atan2(proj(2, :) ./ cos_beta_proj, proj(1, :) ./ cos_beta_proj);
                    uv = real( ...
                    [ ...
                        beta_proj * size(pano_img, 1) / pi + size(pano_img, 1) / 2 + 1; ...
                        alpha_proj * size(pano_img, 2) / (2 * pi) + size(pano_img, 2) / 2 + 1 ...
                    ] ...
                    );

                    % bilinear interpolation from original image
                    img = flip(bilinear_interpolation(img_size, uv, pano_img));

                    if debug
                        figure();
                        imshow(img);
                    end

                    imwrite( ...
                        img, ...
                        reference_img_path, ...
                        'BitDepth', 8 ...
                    );
                end

                % project the factory pointcloud by related projection matrix P
                pcd_rot = x_rot_mat * z_rot_mat * R2';
                P = K * pcd_rot * [eye(3) -C2];

                params_file_path = fullfile(save_root_path, sprintf("%s_params.json", stem));
                if exist(params_file_path, 'file') ~= 2
                    params = struct( ...
                        "x_rot_mat", x_rot_mat, ...
                        "z_rot_mat", z_rot_mat, ...
                        "pano_rot_mat", R2, ...
                        "pano_translation", C2, ...
                        "calibration_mat", K, ...
                        "camera_pose", [pcd_rot' C2; 0 0 0 1], ...
                        "source_pano_path", fullfile(parent, name, 'pano', panorama), ...
                        "source_ply_path", fullfile(parent, name, strcat(name, '_txt.ply')) ...
                    );
                    json_params = jsonencode(params);
                    fid = fopen(params_file_path, 'w');
                    fprintf(fid, json_params);
                    fclose(fid);
                end

                if ~as_colmap
                    [ fpts, frgb ] = filterFieldView( ...
                        struct('R', pcd_rot, 'C', C2), ...
                        pts(1:3,:), pts(4:6,:));
                    uvs = round(h2a(P * a2h(fpts))); % projected points into image plane

                    f = uvs(1,:) > 0 & uvs(2,:) > 0 & uvs(1,:) <= img_size(1) & uvs(2,:) <= img_size(2);
                    uvs = uvs(:, f);
                    frgb = frgb(:, f);

                    pts_projected = uint8(zeros(img_size(2), img_size(1), 3));
                    for i = 1:length(uvs)
                        pts_projected(uvs(2, i), uvs(1, i), :) = frgb(:, i);
                    end

                    if debug
                        figure();
                        imshow(pts_projected);
                    end

                    depth = sqrt(sum((fpts - C2) .^ 2));
                    depth = depth(:, f);
                    pts_depth = zeros(img_size(2), img_size(1), 'double');
                    for i = 1:length(depth)
                        pts_depth(uvs(2, i), uvs(1, i)) = depth(i);
                    end

                    if debug
                        max_depth = max(depth);
                        figure();
                        imshow(pts_depth / max_depth);
                    end

                    imwrite( ...
                        pts_projected, ...
                        fullfile(save_root_path, sprintf("%s_color.png", stem)), ...
                        'BitDepth', 8 ...
                    );

                    imwrite( ...
                        uint16(pts_depth), ...
                        fullfile(save_root_path, sprintf("%s_depth.png", stem)), ...
                        'BitDepth', 16 ...
                    );
                    % Save also real float depth.
                    % savetofile(fullfile(save_root_path, sprintf("%s_depth.mat", stem)), pts_depth);
                else
                    rot_idx = (folder - 3) * pano_count * length(z_rot_angles) * length(x_rot_angles) ...
                        + (pano_id - 1) * length(z_rot_angles) * length(x_rot_angles) ...
                        + (zi - 1) * length(x_rot_angles) + xi;

                    images(rot_idx) = struct(           ...
                        'image_id', rot_idx,            ...
                        'q', r2q(pcd_rot)',             ...
                        'R', pcd_rot,                   ...
                        't', -pcd_rot * C2,             ...
                        'camera_id', 1,                 ...
                        'name', sprintf("%s_reference.png", stem),     ...
                        'xys', zeros(1, 2, 'double'),      ... % uvs',
                        'point3D_ids', zeros(1, 1, 'double') ...  % (length(points3D) + 1):(length(points3D) + size(uvs, 2))
                    );
                end

                fprintf("        FINISHED VIEW %s %d %d\n", ...
                    panorama, z_rot_angles(zi), x_rot_angles(xi));
            end
            fprintf("      FINISHED X STRIDE %s %d\n", ...
                panorama, z_rot_angles(zi));
        end
        fprintf("    FINISHED PANORAMA %s\n", panorama);
    end
    pano_count = pano_count + length(pano_images);
    fprintf("  FINISHED %s\n", name);

    if as_colmap
        fprintf("  SAVING %s to %s\n", name, fullfile(save_root_path, "sparse"));
        saveColmap(fullfile(save_root_path, "sparse"), cameras, images, points3D);
    end

end
fprintf("FINISHED ALL\n");


function savetofile(fullfilename, data)
            save(fullfilename, 'pts_depth');
end
