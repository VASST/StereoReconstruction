/*! \file imageSupport_kernels.cl
 *  \brief Kernels for manipulating images.
 *  \author Nick Lamprianidis
 *  \version 1.1
 *  \date 2015
 *  \copyright The MIT License (MIT)
 *  \par
 *  Copyright (c) 2015 Nick Lamprianidis
 *  \par
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  \par
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  \par
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */


/*! \brief Separates the 3 channels of an RGB image.
 *  \details Performs a matrix transposition on an RGB image `(AoS -> SoA)`.
 *           For avoiding alignment restrictions, the `SoA` structure 
 *           is broken out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] AoS input buffer with the following (logical) arrangement: float[total-pixels][3].
 *                 Each row contains the RGB values of a pixel.
 *  \param[out] r output buffer with all the pixel values in the first channel, R.
 *  \param[out] g output buffer with all the pixel values in the second channel, G.
 *  \param[out] b output buffer with all the pixel values in the third channel, B.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 */
kernel
void separateRGBChannels_Float2Float (global float *AoS, 
                                      global float *r, global float *g, global float *b, 
                                      local float *data)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Each work-item in the work-group reads in a pixel's values
    vstore3 (vload3 (gX, AoS), lX, data);
    barrier (CLK_LOCAL_MEM_FENCE);

    // With each 1/3 work-items in the work-group, indices will offset by one,
    // handling this way first the R, then the G, and then the B values
    uint lastIdx = 3 * lXdim - 1;
    uint baseIdx = (9 * lX) % lastIdx;
    
    // A triplet of values on the same channel
    float3 triplet = { data[baseIdx], data[baseIdx + 3], data[baseIdx + 6] };
    
    // Each 1/3 work-items in the work-group 
    // stores the values of one channel
    uchar channel = (3 * lX) / lXdim;
    global float *img = addr[channel];

    vstore3 (triplet, lX % (lXdim / 3), &img[wgX * lXdim]);
}


/*! \brief Separates the 3 channels of an RGB image.
 *  \details Performs a matrix transposition on an RGB image `(AoS -> SoA)`, 
 *           and promotes the `uchar` type to `float` while normalizing
 *           the values to one. For avoiding alignment restrictions, the `SoA`
 *           structure is broken out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] AoS input buffer with the following (logical) arrangement: uchar[total-pixels][3].
 *                 Each row contains the RGB values of a pixel.
 *  \param[out] r output buffer with all the pixel values in the first channel, R.
 *  \param[out] g output buffer with all the pixel values in the second channel, G.
 *  \param[out] b output buffer with all the pixel values in the third channel, B.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (uchar)` bytes.
 */
kernel
void separateRGBChannels_Uchar2Float (global uchar *AoS, 
                                      global float *r, global float *g, global float *b, 
                                      local uchar *data)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Each work-item in the work-group reads in a pixel's values
    vstore3 (vload3 (gX, AoS), lX, data);
    barrier (CLK_LOCAL_MEM_FENCE);

    // With each 1/3 work-items in the work-group, indices will offset by one,
    // handling this way first the R, then the G, and then the B values
    uint lastIdx = 3 * lXdim - 1;
    uint baseIdx = (9 * lX) % lastIdx;
    
    // A triplet of values on the same channel
    float3 triplet = { data[baseIdx], data[baseIdx + 3], data[baseIdx + 6] };

    // Normalize the values
    triplet /= 255.f;
    
    // Each 1/3 work-items in the work-group 
    // stores the values of one channel
    uchar channel = (3 * lX) / lXdim;
    global float *img = addr[channel];

    vstore3 (triplet, lX % (lXdim / 3), &img[wgX * lXdim]);
}


/*! \brief Combines the 3 channels of an RGB Image.
 *  \details Performs a matrix transposition on an RGB image `(SoA -> AoS)`.
 *           For avoiding alignment restrictions, the `SoA` structure 
 *           is broken out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] r input buffer with all the pixel values in channel R.
 *  \param[in] g input buffer with all the pixel values in channel G.
 *  \param[in] b input buffer with all the pixel values in channel B.
 *  \param[out] AoS output buffer with the following (logical) arrangement: float[total-pixels][3].
 *                  Each row contains the RGB values of a pixel.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 */
kernel
void combineRGBChannels_Float2Float (global float *r, global float *g, global float *b, 
                                     global float *AoS, local float *data)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Each 1/3 work-items in the work-group reads in 
    // a triplet of values on channel, R, G, B, respectively
    uchar channel = (3 * lX) / lXdim;
    uint rank = lX % (lXdim / 3);
    global float *img = addr[channel];
    vstore3 (vload3 (rank, &img[wgX * lXdim]), rank, &data[channel * lXdim]);
    barrier (CLK_LOCAL_MEM_FENCE);

    // Each work-item in the work-group assembles and stores a pixel
    float3 pixel = { data[lX], data[lXdim + lX], data[2 * lXdim + lX] };

    vstore3 (pixel, gX, AoS);
}


/*! \brief Combines the 3 channels of an RGB Image.
 *  \details Performs a matrix transposition on an RGB image `(SoA -> AoS)`,
 *           demotes the `float` type to `uchar`, and scales the data to `255`.
 *           For avoiding alignment restrictions, the `SoA` structure is broken 
 *           out to the individual channels, R, G, B.
 *  \note The global workspace should be one-dimensional `(= # pixels 
 *        in the input buffer)`. The global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] r input buffer with all the pixel values in channel R.
 *  \param[in] g input buffer with all the pixel values in channel G.
 *  \param[in] b input buffer with all the pixel values in channel B.
 *  \param[out] AoS output buffer with the following (logical) arrangement: uchar[total-pixels][3].
 *                  Each row contains the RGB values of a pixel.
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 */
kernel
void combineRGBChannels_Float2Uchar (global float *r, global float *g, global float *b, 
                                     global uchar *AoS, local float *data)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Each 1/3 work-items in the work-group reads in 
    // a triplet of values on channel, R, G, B, respectively
    uchar channel = (3 * lX) / lXdim;
    uint rank = lX % (lXdim / 3);
    global float *img = addr[channel];
    vstore3 (vload3 (rank, &img[wgX * lXdim]), rank, &data[channel * lXdim]);
    barrier (CLK_LOCAL_MEM_FENCE);

    // Each work-item in the work-group assembles and stores a pixel
    float3 triplet = { data[lX], data[lXdim + lX], data[2 * lXdim + lX] };

    // Scale the values
    triplet *= 255.f;

    // Demote the type
    uchar3 pixel = convert_uchar3 (triplet);

    vstore3 (pixel, gX, AoS);
}


/*! \brief Converts a buffer from type `uchort` to `float`.
 *  \note The global workspace should be one dimensional and equal to 
 *        the number of elements in the image divided by 4.
 *
 *  \param[in] depth depth image (for Kinect, type: uint16, unit: mm).
 *  \param[out] fDepth depth image with type `float`.
 *  \param[in] scaling factor by which to scale the depth values in the output array.
 */
kernel
void depth_Ushort2Float (global ushort4 *depth, global float4 *fDepth, float scaling)
{
    uint gX = get_global_id (0);

    fDepth[gX] = convert_float4 (depth[gX]) * scaling;
}


/*! \brief Transforms a depth image to a point cloud.
 *  \note The global workspace should be equal to the dimensions of the image.
 *
 *  \param[in] depth depth image.
 *  \param[out] pCloud point cloud (in 4D homogeneous coordinates with \f$ w=1 \f$).
 *  \param[in] f focal length (for Kinect: 595.f).
 *  \param[in] scaling factor by which to scale the depth values before building the point cloud.
 */
kernel
void depthTo3D (global float *depth, global float4 *pCloud, float f, float scaling)
{
    // Workspace dimensions
    uint cols = get_global_size (0);
    uint rows = get_global_size (1);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);

    // Flatten indices
    uint idx = gY * cols + gX;

    float d = depth[idx] * scaling;
    float4 point = { (gX - (cols - 1) / 2.f) * d / f,  // X = (x - cx) * d / fx
                     (gY - (rows - 1) / 2.f) * d / f,  // Y = (y - cy) * d / fy
                      d, 1.f };                        // Z = d

    pCloud[idx] = point;
}


/*! \brief Performs RGB color normalization.
 *  \details That is $$ \\hat{p}.i = \\frac{p.i}{p.r + p.g + p.b},\\ \\ i=\\{r,g,b\\} $$
 *  \note The global workspace should be one-dimensional `(= # pixels in the input buffer)`.
 *
 *  \param[in] in original frame.
 *  \param[out] out processed frame.
 */
kernel
void rgbNorm (global float *in, global float *out)
{
    uint gX = get_global_id (0);

    // Calculate normalizing factor
    float3 pixel = vload3 (gX, in);
    float sum_ = dot (pixel, 1.f);
    float factor = select (native_recip (sum_), 0.f, isequal (sum_, 0.f));
    
    // Normalize and store
    pixel *= factor;
    vstore3 (pixel, gX, out);
}


/*! \brief Fuses 3-D space coordinates and RGB color values into 
 *         8-D feature points (homogeneous coordinates + RGBA values).
 *  \details Gathers the R, G, B channel values and assembles them into a pixel 
 *           with opacity set to 1.0. It can optionally perform RGB normalization. 
 *           Transforms the depth values into 4-D homogeneous coordinates, setting \f$ w=1 \f$.
 *  \note The global workspace should be one-dimensional `(= # elements 
 *        in the input buffers)`. Both global and local workspaces 
 *        should be **multiples of 3**.
 *
 *  \param[in] depth depth array.
 *  \param[in] r channel R array.
 *  \param[in] g channel G array.
 *  \param[in] b channel B array.
 *  \param[out] p8D array of 8D elements (homogeneous coordinates + RGBA values).
 *  \param[in] data local buffer with size `3 x (# work-items in work-group) x sizeof (float)` bytes.
 *  \param[in] cols number of columns (width) in the input images.
 *  \param[in] f focal length (for Kinect: 595.f).
 *  \param[in] scaling factor by which to scale the depth values before building the point cloud.
 *  \param[in] rgb_norm flag to indicate whether to perform RGB normalization.
 */
kernel
void rgbdTo8D (global float *depth, global float *r, global float *g, global float *b, 
               global float8 *p8D, local float *data, uint cols, float f, float scaling, int rgb_norm)
{
    global float *addr[] = { r, g, b };

    // Workspace dimensions
    uint rows = get_global_size (0) / cols;
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    // Collect RGB values ======================================================

    // Each 1/3 work-items in the work-group reads in 
    // a triplet of values on channel, R, G, B, respectively
    uchar channel = (3 * lX) / lXdim;
    uint rank = lX % (lXdim / 3);
    global float *img = addr[channel];
    vstore3 (vload3 (rank, &img[wgX * lXdim]), rank, &data[channel * lXdim]);
    barrier (CLK_LOCAL_MEM_FENCE);

    // Each work-item in the work-group assembles a pixel    
    float3 pixel = (float3) (data[lX], data[lXdim + lX], data[2 * lXdim + lX]);

    // Perform RGB normalization
    if (rgb_norm)
    {
        float sum_ = dot (pixel, 1.f);
        float factor = select (native_recip (sum_), 0.f, isequal (sum_, 0.f));
        pixel *= factor;
    }
    
    float4 color = (float4) (pixel, 1.f);

    // Build 3D coordinates ====================================================

    uint x = get_global_id (0) % cols;
    uint y = get_global_id (0) / cols;

    float d = depth[gX] * scaling;
    float4 geometry = { (x - (cols - 1) / 2.f) * d / f,  // X = (x - cx) * d / fx
                        (y - (rows - 1) / 2.f) * d / f,  // Y = (y - cy) * d / fy
                         d, 1.f };                       // Z = d

    // Store feature point =====================================================

    float8 point = (float8) (geometry, color);
    p8D[gX] = point;
}


/*! \brief Splits an 8-D point cloud into 4-D geometry points and RGBA color points.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the number 
 *        of points in the point cloud. The local workspace is irrelevant.
 *
 *  \param[in] pc8d array with 8-D points (homogeneous coordinates + RGBA values).
 *  \param[out] pc4d array with 4-D geometry points.
 *  \param[out] rgba array with 4-D color points.
 *  \param[in] offset number of points to skip in the output arrays. The kernel will 
 *                    write in the output arrays starting at position `offset`.
 */
kernel
void splitPC8D (global float8 *pc8d, global float4 *pc4d, global float4 *rgba, unsigned int offset)
{
    uint gX = get_global_id (0);

    float8 point = pc8d[gX];
    size_t pos = offset + gX;
    pc4d[pos] = point.lo;
    rgba[pos] = point.hi;
}


/*! \brief Multiplies two input arrays together, element-wise.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the array, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] a first operand. Input array of `float` elements.
 *  \param[in] b second operand. Input array of `float` elements.
 *  \param[out] out product. Output array of `float` elements.
 */
kernel
void mult (global float4 *a, global float4 *b, global float4 *out)
{
    int gX = get_global_id (0);

	out[gX] = a[gX] * b[gX];
}


/*! \brief Raises an array to an integer power, element-wise.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the array, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] in input array of `float` elements.
 *  \param[out] out output array of `float` elements.
 *  \param[in] n power to which to raise the array.
 */
kernel
void pown_ (global float4 *in, global float4 *out, int n)
{
    int gX = get_global_id (0);

	out[gX] = pown(in[gX], n);
}

/*! \brief Performs an inclusive scan operation on the columns of an array.
 *  \details The parallel scan algorithm by [Blelloch][1] is implemented.
 *           [1]: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 *  \note When there are multiple rows in the array, a scan operation is 
 *        performed per row, in parallel.
 *  \note The number of elements, `N`, in a row of the array should be a **multiple 
 *        of 4** (the data are handled as `float4`). The **x** dimension of the 
 *        global workspace, \f$ gXdim \f$, should be greater than or equal to the number of 
 *        elements in a row of the array divided by 8. That is, \f$ \ gXdim \geq N/8 \f$. 
 *        Each work-item handles `8 float` (= `2 float4`) elements in a row of the array. 
 *        The **y** dimension of the global workspace, \f$ gYdim \f$, should be equal 
 *        to the number of rows, `M`, in the array. That is, \f$ \ gYdim = M \f$. 
 *        The local workspace should be `1` in the **y** dimension, and a 
 *        **power of 2** in the **x** dimension. It is recommended 
 *        to use one `wavefront/warp` per work-group.
 *  \note When the number of elements per row of the array is small enough to be 
 *        handled by a single work-group, the output array will contain the true 
 *        scan result. When the elements are more than that, they are partitioned 
 *        into blocks and scanned independently. In this case, the kernel outputs 
 *        the results from each block scan operation. A scan should then be made on 
 *        the sums of the elements of each block per row. Finally, the results from 
 *        the last block-sums scan should be added in the corresponding block. The 
 *        number of work-groups in the **x** dimension, \f$ wgXdim \f$, **for the 
 *        case of multiple work-groups**, should be made a **multiple of 4**. The 
 *        potential extra work-groups are used for enforcing correctness. They write 
 *        the necessary identity operands, `0`, in the sums array, since in the 
 *        next phase the sums array is going to be handled as `float4`.
 *
 *  \param[in] in input array of `float` elements.
 *  \param[out] out (scan per work-group) output array of `float` elements.
 *  \param[in] data local buffer. Its size should be `2 float` elements for each 
 *                  work-item in a work-group. That is \f$ 2*lXdim*sizeof\ (float) \f$.
 *  \param[out] sums array of block sums. Each work-group outputs the sum of its elements. 
 *                   It's size should be \f$ M \times wgXdim \f$.
 *  \param[in] n the number of elements in a row of the array divided by 4.
 *  \param[in] scaling factor by which to scale the array elements before processing.
 */
kernel
void inclusiveScan_f (global float4 *in, global float4 *out, local float *data, 
                      global float *sums, uint n, float scaling)
{
    // Workspace dimensions
    uint lXdim = get_local_size (0);
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint lX = get_local_id (0);
    uint wgX = get_group_id (0);

    uint offset = 1;

    // Load 8 float elements per work-item
    int4 flag = (int4) (2 * gX) < (int4) (n);
    float4 a = select ((float4) (0.f), in[gY * n + 2 * gX] * scaling, flag);
    flag = (int4) (2 * gX + 1) < (int4) (n);
    float4 b = select ((float4) (0.f), in[gY * n + 2 * gX + 1] * scaling, flag);

    // Perform a serial scan on the 2 float4 elements
    a.y += a.x; a.z += a.y; a.w += a.z;
    b.y += b.x; b.z += b.y; b.w += b.z;

    // Store the sum of each float4 element
    data[2 * lX] = a.w;
    data[2 * lX + 1] = b.w;

    // Perform a scan on the float4 sums

    // Up-Sweep phase
    for (uint d = lXdim; d > 0; d >>= 1)
    {
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
        {
            uint ai = offset * (2 * lX + 1) - 1;
            uint bi = offset * (2 * lX + 2) - 1;
            data[bi] += data[ai];
        }
        offset <<= 1;
    }

    // Store the work-group sum
    if ((wgXdim != 1) && (lX == lXdim - 1))
        sums[gY * wgXdim + wgX] = data[2 * lX + 1];

    // Clear the last register
    if (lX == (lXdim - 1))
        data[2 * lX + 1] = 0.f;

    // Down-Sweep phase
    for (uint d = 1; d < (2 * lXdim); d <<= 1)
    {
        offset >>= 1;
        barrier (CLK_LOCAL_MEM_FENCE);
        if (lX < d)
        {
            uint ai = offset * (2 * lX + 1) - 1;
            uint bi = offset * (2 * lX + 2) - 1;
            float tmp = data[ai];
            data[ai] = data[bi];
            data[bi] += tmp;
        }
    }
    barrier (CLK_LOCAL_MEM_FENCE);

    // Update the sums on the float4 elements
    // and store the results
    if ((2 * gX) < n)
    {
        a += data[2 * lX];
        out[gY * n + 2 * gX] = a;
    }

    if ((2 * gX + 1) < n)
    {
        b += data[2 * lX + 1];
        out[gY * n + 2 * gX + 1] = b;
    }
}


/*! \brief Adds the group sums in the associated blocks.
 *  \details It's the second part of the [Blelloch][1] scan algorithm.
 *           [1]: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 *  \note `scan` handled `2 float4` elements per work-item. `addGroupSums`
 *        handles `1 float4` element per work-item. The global workspace should 
 *        be \f$ 2*(wgXdim-1)*lXdim_{scan} \f$ in the **x** dimension, and \f$ M \f$ 
 *        in the **y** dimension. The global workspace should also have an offset 
 *        \f$ 2*lXdim_{scan} \f$ in the **x** dimension. The local workspace
 *        should be \f$ 2*lXdim_{scan} \f$ in the **x** dimension, and `1`
 *        in the **y** dimension.
 *  \note This part should follow after a scan has been performed on the group sums.
 *
 *  \param[in] sums (scan) array of work-group sums. Its size is \f$M \times wgXdim\f$.
 *  \param[out] out (scan) output array of `float` elements (before processing, it 
 *                  contains the block scans performed in a previous step.
 *  \param[in] n the number of elements in a row of the array divided by 4.
 */
kernel
void addGroupSums_f (global float *sums, global float4 *out, uint n)
{
    // Workspace dimensions
    uint wgXdim = get_num_groups (0);

    // Workspace indices
    uint gX = get_global_id (0);
    uint gY = get_global_id (1);
    uint wgX = get_group_id (0);

    float sum = sums[gY * (wgXdim + 1) + wgX];

    if (gX < n)
        out[gY * n + gX] += sum;
}

/*! \brief Performs a matrix transposition.
 *  \note Both dimensions of the matrix have to be **multiples of 4**. Other 
 *        than that, the matrix can have any dimensions ratio. It doesn't 
 *        have to be square.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of columns, `N`, in the matrix divided by 4. That 
 *        is, \f$ \ gXdim = N/4 \f$. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `M`, in the 
 *        matrix divided by 4. That is, \f$ \ gYdim = M/4 \f$. The local 
 *        workspace should be **square**. That is, \f$ \ lXdim = lYdim \f$.
 *  \note Each work-item transposes a square `4x4` block.
 *
 *  \param[in] in input matrix of `float` elements.
 *  \param[out] out output (transposed) matrix of `float` elements.
 *  \param[in] data local buffer. Its size should be `16 float` elements for 
 *                  each work-item in a work-group. That is, \f$ 4*4*lXdim*lYdim*sizeof\ (float) \f$.
 */
kernel
void transpose (global float4 *in, global float4 *out, local float4 *data)
{
    // Workspace dimensions
    uint gXdim = get_global_size (0);
    uint gYdim = get_global_size (1);
    uint lXdim = get_local_size (0);

    // Workspace indices
    uint lX = get_local_id (0);
    uint lY = get_local_id (1);
    uint wgX = get_group_id (0);
    uint wgY = get_group_id (1);

    uint baseIn = 4 * (wgY * lXdim + lY) * gXdim + (wgX * lXdim + lX);
    uint baseOut = 4 * (wgX * lXdim + lY) * gYdim + (wgY * lXdim + lX);

    // Load a 4x4 block of data within
    // a larger work-group block.
    uint idx = 4 * lY * lXdim + lX;
    data[idx]             = in[baseIn];
    data[idx + lXdim]     = in[baseIn + gXdim];
    data[idx + 2 * lXdim] = in[baseIn + 2 * gXdim];
    data[idx + 3 * lXdim] = in[baseIn + 3 * gXdim];

    barrier (CLK_LOCAL_MEM_FENCE);

    // Read a 4x4 block of data from 
    // the transposed position within 
    // the larger work-group block.
    uint idxTr = 4 * lX * lXdim + lY;
    float4 a = data[idxTr];
    float4 b = data[idxTr + lXdim];
    float4 c = data[idxTr + 2 * lXdim];
    float4 d = data[idxTr + 3 * lXdim];

    // Transpose the 4x4 block of data, and store it 
    // at the same position within the work-group block, 
    // but at the transposed position of the work-group 
    // block within the matrix.
    out[baseOut]             = (float4) (a.x, b.x, c.x, d.x);
    out[baseOut + gYdim]     = (float4) (a.y, b.y, c.y, d.y);
    out[baseOut + 2 * gYdim] = (float4) (a.z, b.z, c.z, d.z);
    out[baseOut + 3 * gYdim] = (float4) (a.w, b.w, c.w, d.w);
}

/*! \brief Performs box (mean) filtering.
 *  \details Accepts a SAT array, \f$ sat_{M \times N} \f$, performs the 
 *           filtering, and outputs the result, \f$ out_{M \times N} \f$.
 *           The work complexity is `O(1)` in the window size.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of columns, `N`, in the image. That is, 
 *        \f$ \ gXdim = N \f$. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `M`, in the 
 *        image. That is, \f$ \ gYdim = M \f$. The local workspace is irrelevant.
 *
 *  \param[in] sat input array of `float` elements.
 *  \param[out] out output (blurred) array of `float` elements.
 *  \param[in] radius radius of the square filter window.
 */
kernel
void boxFilterSAT (global float *sat, global float *out, int radius)
{
    // Workspace dimensions
    int gXdim = get_global_size (0);
    int gYdim = get_global_size (1);

    // Workspace indices
    int gX = get_global_id (0);
    int gY = get_global_id (1);

    // Filter window coordinates
    int2 c0 = { gX - radius - 1, gY - radius - 1 };                            // Top left corner indices
    int2 c1 = { min (gX + radius, gXdim - 1), min (gY + radius, gYdim - 1) };  // Bottom right corner indices
    int2 outOfBounds = isless (convert_float2 (c0), 0.f);

    float sum = 0.f;
    sum += select (sat[c0.y * gXdim + c0.x], 0.f, outOfBounds.x || outOfBounds.y);  // Top left corner
    sum -= select (sat[c0.y * gXdim + c1.x], 0.f, outOfBounds.y);                   // Top right corner
    sum -= select (sat[c1.y * gXdim + c0.x], 0.f, outOfBounds.x);                   // Bottom left corner
    sum +=         sat[c1.y * gXdim + c1.x];                                        // Bottom right corner

    // Number of elements in the filter window
    int2 d = c1 - select (c0, -1, outOfBounds);
    float n = d.x * d.y;

    // Store mean value
    out[gY * gXdim + gX] = sum / n;
}


#define NUM_BF_STORING_WORK_ITEMS 16 * 16 / 4  // 64

/*! \brief Performs box (mean) filtering.
 *  \details Accepts a transposed SAT array, \f$ sat_{N \times M} \f$, performs 
 *           the filtering, and outputs the result, \f$ out_{M \times N} \f$.
 *           The work complexity is `O(1)` in the window size.
 *  \note Both dimensions of the image have to be **multiples of the work-group 
 *        dimensions**, respectively. This specification could be overcome by 
 *        extending the buffers [clEnqueue(Read/Write)BufferRect] and including 
 *        bounds checking within the kernel. These cases won't be handled.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of columns, `M`, in the SAT array. That is, 
 *        \f$ \ gXdim = M \f$. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `N`, in the SAT
 *        array. That is, \f$ \ gYdim = N \f$. The local workspace should be 
 *        `16x16`. That is, \f$ \ lXdim = lYdim = 16 \f$.
 *  \note Each work-item filters one pixel, and then the first 64 work-items in 
 *        each work-group store a transposed 4 pixel block in global memory.

 *  \param[in] sat input array of `float` elements.
 *  \param[out] out output (blurred) array of `float` elements.
 *  \param[in] data local buffer. Its size should be `1 float` element for 
 *                  each work-item in a work-group. That is, \f$ lXdim*lYdim*sizeof\ (float) \f$.
 *  \param[in] radius radius of the square filter window.
 *  \param[in] scaling factor by which to scale the array elements after processing.
 */
kernel
void boxFilterSAT_Tr (global float *sat, global float4 *out, local float *data, int radius, float scaling)
{
    // Workspace dimensions
    int gXdim = get_global_size (0);
    int gYdim = get_global_size (1);
    int lXdim = get_local_size (0);
    int lYdim = get_local_size (1);

    // Workspace indices
    int gX = get_global_id (0);
    int gY = get_global_id (1);
    int lX = get_local_id (0);
    int lY = get_local_id (1);
    int wgX = get_group_id (0);
    int wgY = get_group_id (1);

    // Filter window coordinates
    int2 c0 = { gX - radius - 1, gY - radius - 1 };                            // Top left corner indices
    int2 c1 = { min (gX + radius, gXdim - 1), min (gY + radius, gYdim - 1) };  // Bottom right corner indices
    int2 outOfBounds = isless (convert_float2 (c0), 0.f);

    float sum = 0.f;
    sum += select (sat[c0.y * gXdim + c0.x], 0.f, outOfBounds.x || outOfBounds.y);  // Top left corner
    sum -= select (sat[c0.y * gXdim + c1.x], 0.f, outOfBounds.y);                   // Top right corner
    sum -= select (sat[c1.y * gXdim + c0.x], 0.f, outOfBounds.x);                   // Bottom left corner
    sum +=         sat[c1.y * gXdim + c1.x];                                        // Bottom right corner

    // Number of elements in the filter window
    int2 d = c1 - select (c0, -1, outOfBounds);
    float n = d.x * d.y;

    // Flatten indices
    int idx = lY * lXdim + lX;

    data[idx] = scaling * sum / n;
    barrier (CLK_LOCAL_MEM_FENCE);

    if (idx < NUM_BF_STORING_WORK_ITEMS)
    {
        // Read a transposed float4 element
        //* Elements are processed in column order
        int iy = idx % 4;
        int ix = idx / 4;
        int base = 4 * iy * lXdim + ix;
        float4 pixels = { data[base], 
                          data[base + lXdim], 
                          data[base + 2 * lXdim], 
                          data[base + 3 * lXdim] };

        // Store the float4 element witin the work-group block in the transposed position
        //* Elements are stored in row order. The block has already been transposed
        out[(wgX * lXdim + ix) * gYdim / 4 + (wgY * lYdim / 4 + iy)] = pixels;
    }
}


/*! \brief Performs box (mean) filtering.
 *  \details The work complexity is `O(n)` in the window size.
 *  \note Both dimensions of the image have to be **multiples of the work-group 
 *        dimensions**, respectively. This specification could be overcome by 
 *        extending the buffers [clEnqueue(Read/Write)BufferRect] and including 
 *        bounds checking within the kernel. These cases won't be handled.
 *  \note The **x** dimension of the global workspace, \f$ gXdim \f$, should be 
 *        equal to the number of columns, `N`, in the image. That is, 
 *        \f$ \ gXdim = N \f$. The **y** dimension of the global workspace, 
 *        \f$ gYdim \f$, should be equal to the number of rows, `M`, in the 
 *        image. That is, \f$ \ gYdim = M \f$. The local workspace should be 
 *        `16x16`. That is, \f$ \ lXdim = lYdim = 16 \f$.
 *  \note Vector reads are avoided since the required alignments complicate 
 *        memory address calculations.

 *  \param[in] in input array of `float` elements.
 *  \param[out] out output (blurred) array of `float` elements.
 *  \param[in] data local buffer. Its size should be `1 float` element for 
 *                  each work-item in a work-group and each halo pixel. 
 *                  That is, \f$ (lXdim+2*radius)*(lYdim+2*radius)*sizeof\ (float) \f$.
 *  \param[in] radius radius of the square filter window.
 */
kernel
void boxFilter (global float *in, global float *out, local float *data, int radius)
{
    // Workspace dimensions
    int gXdim = get_global_size (0);
    int gYdim = get_global_size (1);
    int lXdim = get_local_size (0);
    int lYdim = get_local_size (1);
    int lWidth = lXdim + 2 * radius;

    // Workspace indices
    int gX = get_global_id (0);
    int gY = get_global_id (1);
    int lX = get_local_id (0);
    int lY = get_local_id (1);

    // Load data in local memory
    for (int y = lY, iy = gY-radius; y < lYdim + 2 * radius; y += lYdim, iy += lYdim)
    {
        for (int x = lX, ix = gX-radius; x < lXdim + 2 * radius; x += lXdim, ix += lXdim)
        {
            uint flag = (ix >= 0 && iy >= 0 && ix < gXdim && iy < gYdim);
            data[y * lWidth + x] = select (0.f, in[iy * gXdim + ix], flag);
        }
    }
    barrier (CLK_LOCAL_MEM_FENCE);

    // Compute the sum of the filter window elements
    float sum = 0.f;
    for (int fRow = lY; fRow <= lY + 2 * radius; ++fRow)
        for (int fCol = lX; fCol <= lX + 2 * radius; ++fCol)
            sum += data[fRow * lWidth + fCol];

    // Filter window coordinates
    int2 c0 = { gX - radius - 1, gY - radius - 1 };                            // Top left corner indices
    int2 c1 = { min (gX + radius, gXdim - 1), min (gY + radius, gYdim - 1) };  // Bottom right corner indices
    int2 outOfBounds = c0 < 0;

    // Number of elements in the filter window
    int2 d = c1 - select (c0, -1, outOfBounds);
    float n = d.x * d.y;

    // Store mean value
    out[gY * gXdim + gX] = sum / n;
}

/*! \brief Computes the `a` and `b` coefficients in the Guided Filter algorithm.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the arrays, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] mean_p array of average \f$ p \f$ values in the local windows.
 *  \param[in] mean_p2 array of average \f$ p^2 \f$ values in the local windows.
 *  \param[out] a array of \f$ a \f$ coefficients for the local models.
 *  \param[out] b array of \f$ b \f$ coefficients for the local models.
 *  \param[in] eps regularization parameter \f$ \epsilon \f$.
 */
kernel
void gf_ab (global float4 *mean_p, global float4 *mean_p2, 
            global float4 *a, global float4 *b, float eps)
{
    int gX = get_global_id (0);
    
    float4 m_p = mean_p[gX];
    float4 var_p = mean_p2[gX] - pown (m_p, 2);
    float4 a_ = var_p / (var_p + eps);
    
    a[gX] = a_;
    b[gX] = (1.f - a_) * m_p;
}


/*! \brief Computes the filtered output `q` in the Guided Filter algorithm.
 *  \details x.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the arrays, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *  \note When dealing with depth images, there might be invalid pixels
 *        (for Kinect, those pixels have value \f$ 0 \f$). Those invalid pixels 
 *        define surfaces that the Guided Filter algorithm tries to smooth out / 
 *        bring closer to nearby surfaces. The result is that those pixels 
 *        end up all over the place. By setting the `zero_out` flag, a procedure 
 *        is enabled for zeroing out in the \f$ q \f$ array those pixels
 *        that are zero in the \f$ p \f$ array.
 *
 *  \param[in] p input array \f$ p \f$.
 *  \param[in] mean_a array of average \f$ a \f$ values in the local windows.
 *  \param[in] mean_b array of average \f$ b \f$ values in the local windows.
 *  \param[in] q output array \f$ q \f$.
 *  \param[in] zero_out flag to indicate whether to zero out invalid pixels.
 *  \param[in] scaling factor by which to scale the pixel values in the output array.
 */
kernel
void gf_q (global float4 *p, global float4 *mean_a, global float4 *mean_b, 
           global float4 *q, int zero_out, float scaling)
{
    int gX = get_global_id (0);

    float4 p_ = p[gX];
    float4 q_ = mean_a[gX] * p_ + mean_b[gX];

    // Find the zero pixels in p, and if zeroing is enabled,
    // zero out the corresponding pixels in q
    int4 p_select = isequal (p_, 0.f) * zero_out;
    float4 q_z = select(q_, 0.f, p_select);

    q[gX] = scaling * q_z;
}


/*! \brief Computes the `a` and `b` coefficients in the Guided Filter algorithm.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the arrays, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] corr_I array of average \f$ I*I \f$ values in the local windows.
 *  \param[in] corr_Ip array of average \f$ I*p \f$ values in the local windows.
 *  \param[in] mean_I array of average \f$ I \f$ values in the local windows.
 *  \param[in] mean_p array of average \f$ p \f$ values in the local windows.
 *  \param[out] var_I array of variance values for \f$ p \f$ in the local windows.
 *  \param[out] cov_Ip array of covariance values for \f$ I,p \f$ in the local windows.
 */
kernel
void gf_var_Ip (global float4 *corr_I, global float4 *corr_Ip, 
                global float4 *mean_I, global float4 *mean_p, 
                global float4 *var_I, global float4 *cov_Ip)
{
    int gX = get_global_id (0);
    
    float4 m_I = mean_I[gX];

    var_I[gX] = corr_I[gX] - m_I * m_I;
    cov_Ip[gX] = corr_Ip[gX] - m_I * mean_p[gX];
}


/*! \brief Computes the `a` and `b` coefficients in the Guided Filter algorithm.
 *  \note The global workspace should be one-dimensional. The **x** dimension 
 *        of the global workspace, \f$ gXdim \f$, should be equal to the 
 *        number of elements in the arrays, `M x N`, divided by 4. That is, 
 *        \f$ \ gXdim = M*N/4 \f$. The local workspace is irrelevant.
 *
 *  \param[in] var_I array of variance values for \f$ p \f$ in the local windows.
 *  \param[in] cov_Ip array of covariance values for \f$ I,p \f$ in the local windows.
 *  \param[in] mean_I array of average \f$ I \f$ values in the local windows.
 *  \param[in] mean_p array of average \f$ p \f$ values in the local windows.
 *  \param[out] a array of \f$ a \f$ coefficients for the local models.
 *  \param[out] b array of \f$ b \f$ coefficients for the local models.
 *  \param[in] eps regularization parameter \f$ \epsilon \f$.
 */
kernel
void gf_ab_Ip (global float4 *var_I, global float4 *cov_Ip, 
               global float4 *mean_I, global float4 *mean_p, 
               global float4 *a, global float4 *b, float eps)
{
    int gX = get_global_id (0);
    
    float4 a_ = cov_Ip[gX] / (var_I[gX] + eps);
    
    a[gX] = a_;
    b[gX] = mean_p[gX] - a_ * mean_I[gX];
}

/*! \brief Performs RGB to Gray conversion. 
 * \param[in] in input pointer color image data with memory layout  BGR,BGR, BGR,... 
 * \param[out] out output pointer
 */
kernel
void RGB_To_Gray_UcharToFloat(global uchar *in, global float *out)
{
	const int gXdim = get_global_size (0);

	const int gX = get_global_id (0);
	const int gY = get_global_id (1);

	// flattening
	int id = gY*gXdim + gX;
	
	// RGB to Grayscale conversion
	float gray = 0.2989*in[ 3*id+2 ]/* r channel */ + 0.5870*in[ 3*id+1 ]/* g channel*/ + 0.1140*in[ 3*id ];

	// Normalize the output
	out[ id ] = gray/255;
}

/*! \brief Performs computation of the derivative along x direction. 
*  \param[in] p_in input array of float elements.
*  \param[out] out output (derivative) array of `float` elements.
*/
kernel
void x_gradient (global float *p_in, global float *p_out)
{
    // Workspace dimensions
    const int gXdim = get_global_size (0);

    // Workspace indices
    const int gX = get_global_id (0);
    const int gY = get_global_id (1);

    // Filter window coordinates
	int lidx = gX-1;
	int ridx = gX+1;

	// Output
	p_out[ gY*gXdim + gX ] = select( 0.f, .5f*p_in[ gY*gXdim + ridx ] - .5f*p_in[ gY*gXdim + lidx], 
									islessequal (0.f,  (float)lidx) && islessequal ( (float)ridx , (float)gXdim) ) ; 
} 

/*! \brief Perform computation of the derivative along x-y axes. 
 * \param[in] in input array of float elements
 * \param[out] out_x output array of x gradient
 * \param[out] out_y output array of y gradient
 */
kernel
void gradient( global float *in, global float *out_x, global float *out_y)
{
    // Workspace dimensions
    const int gXdim = get_global_size (0);
	const int gYdim = get_global_size (1);

    // Workspace indices
    const int gX = get_global_id (0);
    const int gY = get_global_id (1);

    // Filter window coordinates
	int lidx = gX-1;
	int ridx = gX+1;
	int uidx = gY +1;
	int didx = gY -1;

	// Output ( X gradient )
	out_x[ gY*gXdim + gX ] = select( 0.f, .5f*in[ gY*gXdim + ridx ] - .5f*in[ gY*gXdim + lidx], 
									islessequal (0.f,  (float)lidx) && islessequal ( (float)ridx , (float)gXdim) ) ; 

	// Output ( Y gradient )
	out_y[ gY*gXdim + gX ] = select( 0.f, .5f*in[ uidx*gXdim + gX ] - .5f*in[ didx*gXdim + gX], 
									islessequal (0.f,  (float)didx) && islessequal ( (float)uidx , (float)gYdim) ) ; 

}


/*! \brief Performs sobel filtering along x and y directions.
 * \param[in] src input array of float elements.
 * \param[out] dst output array of 'float' elements. 
 * \param[in] mode input argument specifying whether to compute Gx, Gy, or G = sqrt(Gx.^2+Gy.^2)
 */
kernel
void sobel(global float *src, global float *dst, int mode)
{
    // Workspace dimensions
    const int gXdim = get_global_size (0);

    // Workspace indices
    const int gX = get_global_id (0);
    const int gY = get_global_id (1);

	switch( mode )
	{
	case 0:
		break;

	case 1:
		break;

	case 2:
		break;
	}

	dst[ gY*gXdim + gX ] = 0.f;
}

/*! \brief Performs a RGB to gray conversion
 * \param[in] color float3 vector of color
 * \return float of gray value
 */
float convert_to_gray(float3 color)
{
	return (0.2989*color.x + 0.5870*color.y + 0.1140*color.z);
}

/*! \brief Performs color cost computation per work-item
 * \param[in] colorL vector of 3 colors of the left image
 * \param[in] colorR vectgor of 3 colors of the right image
 * \param[in] MAX_COST cost threshold
 */

float brightness_cost( float I_L, float I_R, float MAX_COST)
{
	// Refer to this article for better color differences https://en.wikipedia.org/wiki/Color_difference
	// Use Lab color space instread for improved performance. 
	float cost = fabs(I_L - I_R); 
	return select( cost, MAX_COST, isgreater( cost, MAX_COST ) );
}

/*! \brief Performs gradient cost computation per work-item
 * \param[in] gradL floating point gradient value of the left image
 * \pram[in] gradR floating point gradient value of the right image
 */
float grad_cost( float gradL, float gradR, float MAX_COST )
{
	float cost = fabs( gradL - gradR );
	return select( cost, MAX_COST, isgreater( cost, MAX_COST ) );
}


/*! \brief Performs computation of cost volume . 
 * \param[in] Lr input array of r channel of the left image
 * \param[in] Lg input array of g channel of the left image
 * \param[in] Lb input array of b channel of the left image
 * \param[in] Rr input array of r channel of the right image
 * \param[in] Rg input array of g channel of the right image
 * \param[in] Rb input array of b channel of the right image
 * \param[in] gradL input array of the gradient of the left image
 * \param[in] gradR input array of the gradient of the right image
 * \param[out] out output array of the computed costs at different disparities. Is of size width*height*d
 * \param[in] patch_r radius of an image patch
 * \param[in] in d input levels of disparities
 * \param[in] color_th input color threshold
 * \param[in] grad_th input gradient threshold
 */
kernel
void compute_cost(global float *L, global float *R, 
					 global float *gradL, global float *gradR, global float *out, 
					 int patch_r, 
					 int d_min, int d_max, float color_th, float grad_th, float alpha, 
					 int similarity, int reference_img , float inf)
{
    // Workspace dimensions
    const int gXdim = get_global_size (0);
	const int gYdim = get_global_size (1);
	const int gZdim = get_global_size (2); // disparity direction

    // Workspace indices
    const int gX = get_global_id (0);
    const int gY = get_global_id (1);
	const int gZ = get_global_id (2); // disparity direction

	// volume id
	const int vid = gZ*gXdim*gYdim + gY*gXdim + gX; 
	// Set output to 0
	out[ vid ] = 0.0;
	
	// work-group ids
	const int wid = gY*gXdim + gX; 

	// disparity 
	const int d = d_min + gZ;
			
	float data_term = (1-alpha)*color_th + alpha*grad_th;
	
	// NOTE: For a rectified image-pair, the disparity ranges from 0-d_max. The corresponding pixel in the right
	// image always occur towards the left of the pixel in the left image. So, (IL, IR-d)
	if( 0 <= gX-d )
	{

		int vx = 2*(patch_r-1)+1;

		if( patch_r <=gX && gX<=gXdim-patch_r-1 && patch_r <gY && gY<=gYdim-patch_r-1) // check bounds
		{

			// Compute ZNCC
			int j, k, l, img_idx;
			float I_L0 = 0.f;
			float I_R0 = 0.f;
			float sumL = 0.f;
			float sumR = 0.f;
			float prod_sum = 0.f;
			float square_sum_L = 0.f;
			float square_sum_R = 0.f;

			int N = vx*vx;
			for( int i=0; i< N; i++)
			{
				j = i%vx - patch_r +1;
				k = i/vx - patch_r +1;
				img_idx = wid + k*gXdim + j;

				float _I_L = L[img_idx];
				float _I_R = R[img_idx - d ];

				sumL += _I_L;
				sumR += _I_R;

				prod_sum += _I_L * _I_R; // v0*vn
				square_sum_L += _I_L * _I_L; // v0*v0
				square_sum_R += _I_R * _I_R; // vn*vn
			}
			I_L0 = sumL/N;   I_R0 = sumR/N;
		
			float num = (prod_sum - N * I_L0 * I_R0);
			float den = (square_sum_L - sumL*sumL/N)*(square_sum_R - sumR*sumR/N);

			// Similarity measures
			float zncc = num/sqrt(den);	
			float absolute_diff = 0.f;
			float gradient_cost = 0.f;

			switch( reference_img )
			{
			case 1:
				absolute_diff   = brightness_cost( L[ wid ], R[ wid - d ], color_th );
				gradient_cost   = grad_cost( gradL[ wid ], gradR[ wid - d ], grad_th);
				break;

			case 2:
				absolute_diff   = brightness_cost( L[ wid ], R[ wid + d ], color_th );
				gradient_cost   = grad_cost( gradL[ wid ], gradR[ wid + d ], grad_th);
			}

			switch( similarity )
			{
				case 0:
					// Zero-mean Normalized Cross Correlation
					data_term = zncc; 
					break;

				case 1:
					// AD
					data_term = absolute_diff; 
					break;

				case 2:
					// Cost Eq. (5) Hosni et. al. PAMI 2011.
					data_term = ( 1 - alpha )*absolute_diff + alpha*gradient_cost;
					break;
			}

		} 
		out[ vid ] = data_term;
	}
}


/*! \brief Performs WTA optimization on a width*height*d volume
 * \param[in] in input buffer
 * \pram[out[ out output buffer
 */
kernel 
void WTA_Optimizer(global float *in, global float *out, int d_min,int n)
{
	// Workspace dimensions
    const int gXdim = get_global_size (0);
	const int gYdim = get_global_size (1);

    // Workspace indices
    const int gX = get_global_id (0);
    const int gY = get_global_id (1);

	const int d_max = n + d_min +1;

	if( 0 <= gX-d_max )
	{
		float cost = 100000.f;
		// WTA
		for(int i=0; i<n; i++)
		{
			float data = in[ i*gXdim*gYdim + gY*gXdim + gX ];
			if( cost >= data  )
			{
				cost = data;
				out[gY*gXdim + gX] = (float)(d_min + i);
			}
		}
	}
	else
		out[gY*gXdim + gX] = 0.f;
}

// CL kernels for Chang et. al. ,MICCAI 2013
// Diffusion tensor D = exp( -alpha*(nabla(I))^beta)
/*! \brief Compute the Diffusion tensor for a given image
 * \param[in] img_grad_x pointer to the X gradient of the image. 
 * \param[[in] img_grad_y pointer to the Y gradient of the image. 
 * \param[out] tensor output tensor for diffusion tensors stored in the following order [Dxx Dxy Dyx Dyy] .. 
 *             So this array is 4*width*height long. 
 */
kernel
void DiffuseTensor(global float *img, global float *tensor, float alpha, float beta, int min_d, int max_d, int radius)
{
	// Workspace dimensions
    const int gXdim = get_global_size (0);
	const int gYdim = get_global_size (1);

    // Workspace indices
    const int gX = get_global_id (0);
    const int gY = get_global_id (1);

	if( radius < gY && gY < gYdim-radius && gX < gXdim+min_d-radius && max_d+radius < gX)
	{
		float nabla_img[2];
		float current_data = img[ gY*gXdim + gX ];
		float x_forward_data = img[ gY*gXdim + gX + 1 ];
		float y_forward_data = img[ (gY+1)*gXdim + gX ];

		// Take forward differences. ( Ref. Werlberger et. al., BMCV 2009 )
		nabla_img[0] = x_forward_data - current_data;
		nabla_img[1] = y_forward_data - current_data;

		float norm_nabla_img = sqrt( nabla_img[0]*nabla_img[0] + nabla_img[1]*nabla_img[1] );

		float edge_weight = exp( -alpha*pow(norm_nabla_img, beta));

		if( norm_nabla_img > 0 )
		{
			float n[2], p_n[2];

			n[0] = nabla_img[0]/norm_nabla_img;
			n[1] = nabla_img[1]/norm_nabla_img;
			p_n[0] = -n[1];
			p_n[1] = n[0];

			/* Diffusion tensor indexed as [0 2; 1 3] */
			int id = gY*gXdim + gX;
			tensor[ id ]   = edge_weight*(n[0]*n[0]) + p_n[0]*p_n[0];
			tensor[ id+1 ] = edge_weight*(n[1]*n[0]) + p_n[1]*p_n[0];
			tensor[ id+2 ] = edge_weight*(n[0]*n[1]) + p_n[0]*p_n[1];
			tensor[ id+3 ] = edge_weight*(n[1]*n[1]) + p_n[1]*p_n[1];
		}
		else
		{
			int id = gY*gXdim + gX;
			tensor[ id ]   = edge_weight;
			tensor[ id+1 ] = 0.f;
			tensor[ id+2 ] = 0.f;
			tensor[ id+3 ] = edge_weight; 
		}

	}
}

/*! \brief Diffusion Preconditioning
 * \param[in] diffusion_tensor input pointer to Diffusion Tensor array 
 */
kernel 
void DiffusionPreconditioning(global float *diffusion_tensor, global float *dual_step_sigma0, global float *dual_step_sigma1, 
										global float *primal_step_tau, 
											int min_d, int max_d, int radius)
{
	// Workspace dimensions
    const int gXdim = get_global_size (0);
	const int gYdim = get_global_size (1);

    // Workspace indices
    const int gX = get_global_id (0);
    const int gY = get_global_id (1);

	if( gY < gYdim-radius && radius < gY && gX < gXdim+min_d-radius && max_d+radius < gX )
	{
		float D[4], Dx[4], Dy[4];

		const int idx = gY*gXdim + gX;

		D[0] = diffusion_tensor[ idx ];
		D[1] = diffusion_tensor[ idx+1 ];
		D[2] = diffusion_tensor[ idx+2 ];
		D[3] = diffusion_tensor[ idx+3 ];

		Dx[0] = diffusion_tensor[ idx-4 ];
		Dx[1] = diffusion_tensor[ idx-3 ];
		Dx[2] = diffusion_tensor[ idx-2 ];
		Dx[3] = diffusion_tensor[ idx-1 ];

		Dy[0] = diffusion_tensor[ idx - gXdim ];
		Dy[1] = diffusion_tensor[ idx - gXdim +1 ];
		Dy[2] = diffusion_tensor[ idx - gXdim +2 ];
		Dy[3] = diffusion_tensor[ idx - gXdim +3 ];

		/* D*nabla sum */
		if( gX < gXdim-radius-1)
			dual_step_sigma0[  idx ] = 2*fabs(D[0]) + 2*fabs(D[2]);

		if( gY < gYdim -radius -1)
			dual_step_sigma1[ idx ]  = 2*fabs(D[1]) + 2*fabs(D[3]);

		float val = dual_step_sigma0[ idx ]; 

		dual_step_sigma0[ idx ] = select( 1.f/val , 0.f, val == 0.f); 
		dual_step_sigma1[ idx ] = select( 1.f/val, 0.f, val == 0.f); 

		/* D*nabla sum */
		primal_step_tau[ idx ] += fabs(D[0]) + fabs(D[2]);
		primal_step_tau[ idx ] += fabs(D[1]) + fabs(D[3]);

		if( gX > radius && gY > max_d + radius)
			primal_step_tau[ idx ] += fabs(Dx[0]) + fabs(Dx[3]);

		if( gX == radius && gY < max_d+radius)
			primal_step_tau[ idx ] += fabs(Dy[3]);

		if( gY == max_d + radius && gX > radius)
			primal_step_tau[ idx ] += fabs(Dx[0]);
		
		val = primal_step_tau[ idx ];
		primal_step_tau[ idx ] =select(1.f/val , 0.f,val == 0.f); 
	}

}
		

/*! \brief Huber-L2 Preconditioning dual updating kernel
 * \param[in] 
 */
kernel
void HuberL2DualUpdate(global float *diffusion_tensor, global float *head_primal, global float *dual_step_sigma0, global float *dual_step_sigma1,
								global float *dual_p0, global float *dual_p1, float epsilon, int max_d, int radius)
{
	// Workspace dimensions
    const int gXdim = get_global_size (0);
	const int gYdim = get_global_size (1);

    // Workspace indices
    const int gX = get_global_id (0);
    const int gY = get_global_id (1);

	if( gX < gXdim-radius && radius <= gX && gY < gYdim-radius && max_d+radius <= gY )
	{
		float D_nabla_primal[2], sigma[2], D[4];

		int idx = gY*gXdim + gX;
		
		D[0] = diffusion_tensor[ idx ];
		D[1] = diffusion_tensor[ idx+1 ];
		D[2] = diffusion_tensor[ idx+2 ];
		D[3] = diffusion_tensor[ idx+3 ];

		if( gX < gXdim-radius-1 )
			D_nabla_primal[0] = -(D[0]+D[2])*head_primal[ idx ] +
									D[0]*head_primal[ idx + 1 ] + D[2]*head_primal[ idx + gXdim]; 

		if( gY < gYdim-radius-1)
			D_nabla_primal[1] = -(D[1]+D[3])*head_primal[ idx ] +
									D[1]*head_primal[ idx + 1 ] + D[3]*head_primal[ idx + gXdim ];

		if( gX == gXdim-radius-1)
			D_nabla_primal[1] = -(D[1]+D[3])*head_primal[ idx ] + D[3]*head_primal[ idx + gXdim ];

		if( gY == gYdim-radius-1)
			D_nabla_primal[0] = -(D[0]+D[2])*head_primal[ idx ] + D[0]*head_primal[ idx + 1 ];

		sigma[0] = dual_step_sigma0[ idx ];
		sigma[1] = dual_step_sigma1[ idx ];

		
		dual_p0[ idx ] = (dual_p0[ idx ] + sigma[0]*D_nabla_primal[0])/(1.f + sigma[0]*epsilon);
		dual_p1[ idx ] = (dual_p1[ idx ] + sigma[1]*D_nabla_primal[1])/(1.f + sigma[1]*epsilon);

		// constrain ||p|| <= 1.0
		float p_norm = sqrt( dual_p0[ idx ]* dual_p0[ idx ] + dual_p1[ idx ]*dual_p1[ idx ]);
		float reprojection = max( 1.0f, p_norm);

		dual_p0[ idx ] /= reprojection;
		dual_p1[ idx ] /= reprojection;
	}
}