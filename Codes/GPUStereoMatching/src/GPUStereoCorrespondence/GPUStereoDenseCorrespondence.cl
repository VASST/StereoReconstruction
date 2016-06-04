/*! \brief Performs computation of the derivative along x direction. 
*  \param[in] p_in input array of float elements.
*  \param[out] out output (derivative) array of `float` elements.
*/
/*kernel
void x_gradient (global float *p_in, global float *p_out)
{
    // Workspace dimensions
 /*   int gXdim = get_global_size (0);
    int gYdim = get_global_size (1);

    // Workspace indices
    int gX = get_global_id (0);
    int gY = get_global_id (1);

    // Filter window coordinates
	int lidx = gX-1;
	int ridx = gX+1;
    int outOfBounds = isless ( lidx, 0.f) | isless ( ridx , gXdim);

	p_out[ gY*gXdim + gX ] = select( .5f*p_in[ gY*gXdim + lidx ] + .5f*p_in[ gY*gXdim + ridx], 0.f, outofBouds ); 
} */

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