/*
 * Copyright (c) 2017 Juliette Foucaut & Doug Binks
 * 
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 * 
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 * 
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */
#pragma once

#include <stdint.h>
#include "miniz.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENKI_MI_REGION_CHUNKS_NUMBER 1024

// http://web.archive.org/web/20110723210920/http://www.minecraft.net/docs/NBT.txt
typedef enum
{
	enkiNBTTAG_End = 0,
	enkiNBTTAG_Byte = 1,
	enkiNBTTAG_Short = 2,
	enkiNBTTAG_Int = 3,
	enkiNBTTAG_Long = 4,
	enkiNBTTAG_Float = 5,
	enkiNBTTAG_Double = 6,
	enkiNBTTAG_Byte_Array = 7,
	enkiNBTTAG_String = 8,
	enkiNBTTAG_List = 9,
	enkiNBTTAG_Compound = 10,
	enkiNBTTAG_Int_Array = 11,
} enkiNBTTAG_ID;


typedef struct enkiNBTTagHeader_s
{
	char* pName;

	// if the tag is a list, we need the following variables
	int32_t listNumItems;
	int32_t listCurrItem;
	uint8_t listItemTagId;

	// the tagId of type enkiNBTTAG_ID
	uint8_t tagId;
} enkiNBTTagHeader;


const char* enkiGetNBTTagIDAsString( uint8_t tagID_ );
const char* enkiGetNBTTagHeaderIDAsString( enkiNBTTagHeader tagID_ );

typedef struct enkiNBTDataStream_s
{
	enkiNBTTagHeader parentTags[ 512 ];
	enkiNBTTagHeader currentTag;
	uint8_t* pCurrPos;
	uint8_t* pDataEnd;
	uint8_t* pData;
	uint8_t* pNextTag;
	uint8_t* pAllocation;
	uint32_t dataLength;
	int32_t level;
} enkiNBTDataStream;


// Initialize stream from memory pointer.
// pData_ and it's contents should remain valid until 
// after enkiNBTDataStream no longer needed.
// Contents of buffer will be modified for easier reading,
// namely tag name strings will moved down a byte, null terminated,
// and prefixed with 0xFF instead of int16_t string length.
// Make a copy if you need to use the buffer in another lib.
// Other strings in file will not be altered.
// pUnCompressedData_ should be freed by caller.
// FreeMemoryAllocated() should still be called to free any internal allocations.
void enkiNBTInitFromMemoryUncompressed( enkiNBTDataStream* pStream_, uint8_t* pUnCompressedData_, uint32_t dataSize_ );

// Initialize stream from memory pointer to compressed content.
// This function will allocate space for uncompressed stream and decompress it with zlib.
// If uncompressedSizeHint_ > compressedDataSize_ it will be used as the starting hint size for allocating
// the uncompressed size.
// returns 1 if successfull, 0 if not.
int enkiNBTInitFromMemoryCompressed( enkiNBTDataStream* pStream_, uint8_t* pCompressedData_,
									    uint32_t compressedDataSize_, uint32_t uncompressedSizeHint_ );


// returns 0 if no next tag, 1 if there was
int enkiNBTReadNextTag( enkiNBTDataStream* pStream_ );


// Rewind stream so it can be read again from beginning
void enkiNBTRewind( enkiNBTDataStream* pStream_ );

// Frees any internally allocated memory.
void enkiNBTFreeAllocations( enkiNBTDataStream* pStream_ );

int8_t  enkiNBTReadInt8(   enkiNBTDataStream* pStream_ );
int8_t  enkiNBTReadByte(   enkiNBTDataStream* pStream_ );
int16_t enkiNBTReadInt16(  enkiNBTDataStream* pStream_ );
int16_t enkiNBTReadShort(  enkiNBTDataStream* pStream_ );
int32_t enkiNBTReadInt32(  enkiNBTDataStream* pStream_ );
int32_t enkiNBTReadInt(    enkiNBTDataStream* pStream_ );
float   enkiNBTReadFloat(  enkiNBTDataStream* pStream_ );
int64_t enkiNBTReadInt64(  enkiNBTDataStream* pStream_ );
int64_t enkiNBTReadlong(   enkiNBTDataStream* pStream_ );
double  enkiNBTReadDouble( enkiNBTDataStream* pStream_ );

typedef struct enkiNBTString_s
{
	int16_t     size;
	const char* pStrNotNullTerminated;
} enkiNBTString;

enkiNBTString enkiNBTReadString( enkiNBTDataStream* pStream_ );

typedef struct enkiRegionFile_s
{
	uint8_t* pRegionData;
	uint32_t regionDataSize;
} enkiRegionFile;

// enkiRegionFileInit simply zeros data
void enkiRegionFileInit( enkiRegionFile* pRegionFile_ );

enkiRegionFile enkiRegionFileLoad( FILE* fp_ );

void enkiInitNBTDataStreamForChunk( enkiRegionFile regionFile_, int32_t chunkNr_, enkiNBTDataStream* pStream_ );

int32_t enkiGetTimestampForChunk( enkiRegionFile regionFile_, int32_t chunkNr_ );

// enkiFreeRegionFileData frees data allocated in enkiRegionFile
void enkiRegionFileFreeAllocations( enkiRegionFile* pRegionFile_ );

// Check if lhs_ and rhs_ are equal, return 1 if so, 0 if not.
// Safe to pass in NULL for either
// Note that both NULL gives 0.
int enkiAreStringsEqual( const char* lhs_, const char* rhs_ );

#define ENKI_MI_NUM_SECTIONS_PER_CHUNK 16
#define ENKI_MI_SIZE_SECTIONS 16

typedef struct enkiMICoordinate_s
{
	int32_t x;
	int32_t y; // height
	int32_t z;
} enkiMICoordinate;

typedef struct enkiChunkBlockData_s
{
	uint8_t* sections[ ENKI_MI_NUM_SECTIONS_PER_CHUNK ];
	int32_t xPos; // section coordinates
	int32_t zPos; // section coordinates
	int32_t countOfSections;
} enkiChunkBlockData;

// enkiChunkInit simply zeros data
void enkiChunkInit( enkiChunkBlockData* pChunk_ );

// enkiNBTReadChunk gets a chunk from an enkiNBTDataStream
// No allocation occurs - section data points to enkiNBTDataStream.
// pStream_ mush be kept valid whilst chunk is in use.
enkiChunkBlockData enkiNBTReadChunk( enkiNBTDataStream* pStream_ );


enkiMICoordinate enkiGetChunkOrigin( enkiChunkBlockData* pChunk_ );

// get the origin of a section (0 <-> ENKI_MI_NUM_SECTIONS_PER_CHUNK).
enkiMICoordinate enkiGetChunkSectionOrigin( enkiChunkBlockData* pChunk_, int32_t section_ );

// sectionOffset_ is the position from enkiGetChunkSectionOrigin
// Performs no safety checks.
// check pChunk_->sections[ section_ ] for NULL first in your code.
// and ensure sectionOffset_ coords with 0 to ENKI_MI_NUM_SECTIONS_PER_CHUNK
uint8_t enkiGetChunkSectionVoxel( enkiChunkBlockData* pChunk_, int32_t section_, enkiMICoordinate sectionOffset_ );

uint32_t* enkiGetMineCraftPalette(); //returns a 256 array of uint32_t's in uint8_t rgba order.

#ifdef __cplusplus
};
#endif
