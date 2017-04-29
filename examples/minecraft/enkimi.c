
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

#include "enkimi.h"


static const uint32_t SECTOR_SIZE = 4096;

static const char* tagIdString[] =
{
	"TAG_End",
	"TAG_Byte",
	"TAG_Short",
	"TAG_Int",
	"TAG_Long",
	"TAG_Float",
	"TAG_Double",
	"TAG_Byte_Array",
	"TAG_String",
	"TAG_List",
	"TAG_Compound",
	"TAG_Int_Array",
};

static uint32_t minecraftPalette[] = 
{ 
	0xff000000, 0xff7d7d7d, 0xff4cb376, 0xff436086, 0xff7a7a7a, 0xff4e7f9c, 0xff256647, 0xff535353, 0xffdcaf70, 0xffdcaf70, 
	0xff135bcf, 0xff125ad4, 0xffa0d3db, 0xff7a7c7e, 0xff7c8b8f, 0xff7e8287, 0xff737373, 0xff315166, 0xff31b245, 0xff54c3c2, 
	0xfff4f0da, 0xff867066, 0xff894326, 0xff838383, 0xff9fd3dc, 0xff324364, 0xff3634b4, 0xff23c7f6, 0xff7c7c7c, 0xff77bf8e, 
	0xffdcdcdc, 0xff296595, 0xff194f7b, 0xff538ba5, 0xff5e96bd, 0xffdddddd, 0xffe5e5e5, 0xff00ffff, 0xff0d00da, 0xff415778, 
	0xff0d0fe1, 0xff4eecf9, 0xffdbdbdb, 0xffa1a1a1, 0xffa6a6a6, 0xff0630bc, 0xff0026af, 0xff39586b, 0xff658765, 0xff1d1214, 
	0xff00ffff, 0xff005fde, 0xff31271a, 0xff4e87a6, 0xff2a74a4, 0xff0000ff, 0xff8f8c81, 0xffd5db61, 0xff2e5088, 0xff17593c, 
	0xff335682, 0xff676767, 0xff00b9ff, 0xff5b9ab8, 0xff387394, 0xff345f79, 0xff5190b6, 0xff6a6a6a, 0xff5b9ab8, 0xff40596a, 
	0xff7a7a7a, 0xffc2c2c2, 0xff65a0c9, 0xff6b6b84, 0xff2d2ddd, 0xff000066, 0xff0061ff, 0xff848484, 0xfff1f1df, 0xffffad7d, 
	0xfffbfbef, 0xff1d830f, 0xffb0a49e, 0xff65c094, 0xff3b5985, 0xff42748d, 0xff1b8ce3, 0xff34366f, 0xff334054, 0xff45768f, 
	0xffbf0a57, 0xff2198f1, 0xffffffec, 0xffb2b2b2, 0xffb2b2b2, 0xffffffff, 0xff2d5d7e, 0xff7c7c7c, 0xff7a7a7a, 0xff7cafcf, 
	0xff78aaca, 0xff6a6c6d, 0xfff4efd3, 0xff28bdc4, 0xff69dd92, 0xff53ae73, 0xff0c5120, 0xff5287a5, 0xff2a4094, 0xff7a7a7a, 
	0xff75718a, 0xff767676, 0xff1a162c, 0xff1a162c, 0xff1a162c, 0xff2d28a6, 0xffb1c454, 0xff51677c, 0xff494949, 0xff343434, 
	0xffd18934, 0xffa5dfdd, 0xff0f090c, 0xff316397, 0xff42a0e3, 0xff4d84a1, 0xff49859e, 0xff1f71dd, 0xffa8e2e7, 0xff74806d, 
	0xff3c3a2a, 0xff7c7c7c, 0xff5a5a5a, 0xff75d951, 0xff345e81, 0xff84c0ce, 0xff455f88, 0xff868b8e, 0xffd7dd74, 0xff595959, 
	0xff334176, 0xff008c0a, 0xff17a404, 0xff5992b3, 0xffb0b0b0, 0xff434347, 0xff1d6b9e, 0xff70fdfe, 0xffe5e5e5, 0xff4c4a4b, 
	0xffbdc6bf, 0xffddedfb, 0xff091bab, 0xff4f547d, 0xff717171, 0xffdfe6ea, 0xffe3e8eb, 0xff41819b, 0xff747474, 0xffa1b2d1, 
	0xfff6f6f6, 0xff878787, 0xff395ab0, 0xff325cac, 0xff152c47, 0xff65c878, 0xff3534df, 0xffc7c7c7, 0xffa5af72, 0xffbec7ac, 
	0xff9fd3dc, 0xffcacaca, 0xff425c96, 0xff121212, 0xfff4bfa2, 0xff1474cf, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xff1d56ac, 
	0xff1d57ae, 0xff1d57ae, 0xff1d57ae, 0xff243c50, 0xff8dcddd, 0xff4d7aaf, 0xff0e2034, 0xff366bcf, 0xff355d7e, 0xff7bb8c7, 
	0xff5f86bb, 0xff1e2e3f, 0xff3a6bc5, 0xff30536e, 0xffe0f3f7, 0xff5077a9, 0xff2955aa, 0xff21374e, 0xffcdc5dc, 0xff603b60, 
	0xff856785, 0xffa679a6, 0xffaa7eaa, 0xffa879a8, 0xffa879a8, 0xffa879a8, 0xffaae6e1, 0xffaae6e1, 0xff457d98, 0xfff0f0f0, 
	0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 
	0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 
	0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 
	0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 
	0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xfff0f0f0, 0xff242132 };


typedef struct SectionChunkInfo_s
{
	uint8_t offset_0;
	uint8_t offset_1;
	uint8_t offset_2;
	uint8_t sectorCount;
} SectionChunkInfo;

static int32_t GetChunkLocation( SectionChunkInfo section_ )
{
	return ( ( ( section_.offset_0 << 16 ) + ( section_.offset_1 << 8 ) + section_.offset_2 ) * SECTOR_SIZE );
}

typedef struct BigEndian4BytesTo32BitInt_s
{
	uint8_t pos_0;
	uint8_t pos_1;
	uint8_t pos_2;
	uint8_t pos_3;
} BigEndian4BytesTo32BitInt;


static int32_t Get32BitInt( BigEndian4BytesTo32BitInt in_ )
{
	return ( ( in_.pos_0 << 24 ) + ( in_.pos_1 << 16 ) + ( in_.pos_2 << 8 ) + in_.pos_3 );
}

typedef struct RegionHeader_s
{
	SectionChunkInfo          sectionChunksInfos[ ENKI_MI_REGION_CHUNKS_NUMBER ]; // chunks locations
	BigEndian4BytesTo32BitInt sectionChunksTimestamps[ ENKI_MI_REGION_CHUNKS_NUMBER ]; // chunks timestamps
} RegionHeader;

const char * enkiGetNBTTagIDAsString( uint8_t tagID_ )
{
	return tagIdString[ tagID_ ];
}

const char * enkiGetNBTTagHeaderIDAsString( enkiNBTTagHeader tagheader_ )
{
	return tagIdString[ tagheader_.tagId ];
}

void enkiNBTInitFromMemoryUncompressed( enkiNBTDataStream* pStream_, uint8_t * pData_, uint32_t dataSize_)
{
	memset( &pStream_->currentTag, 0, sizeof( enkiNBTTagHeader ) );
	pStream_->pData = pData_;
	pStream_->dataLength = dataSize_;
	pStream_->pCurrPos = pStream_->pData;
	pStream_->pDataEnd = pStream_->pData + pStream_->dataLength;
	pStream_->pNextTag = pStream_->pCurrPos;
	pStream_->level = -1;
	pStream_->pAllocation = NULL;
}

int enkiNBTInitFromMemoryCompressed( enkiNBTDataStream* pStream_, uint8_t * pCompressedData_,
										uint32_t compressedDataSize_, uint32_t uncompressedSizeHint_)
{
	mz_ulong destLength = uncompressedSizeHint_;
	if( destLength <= compressedDataSize_ )
	{
		destLength = compressedDataSize_ * 4 + 1024; // estimate uncompressed size
	}
	mz_ulong startDestLength = destLength;
	uint8_t* dataUnCompressed = (uint8_t*)malloc( destLength );
	int retval = uncompress( dataUnCompressed, &destLength, pCompressedData_, compressedDataSize_ );
	if( retval == MZ_BUF_ERROR && startDestLength == destLength )
	{
		// failed to uncompress, buffer full
		for( int attempts = 0; ( retval != MZ_OK ) && ( attempts < 3 ); ++attempts )
		{
			free( dataUnCompressed );
			destLength *= 4 + 1024;
			dataUnCompressed = (uint8_t*)malloc( destLength );
			retval = uncompress( dataUnCompressed, &destLength, pCompressedData_, compressedDataSize_ );
		}
	}
	if( retval != MZ_OK )
	{
		enkiNBTInitFromMemoryUncompressed( pStream_, NULL, 0 );
		free( dataUnCompressed );
		return 0;
	}

	dataUnCompressed = (uint8_t*)realloc( dataUnCompressed, destLength ); // reallocate to actual size
	enkiNBTInitFromMemoryUncompressed( pStream_, dataUnCompressed, ( uint32_t )destLength );
	pStream_->pAllocation = dataUnCompressed;
	return 1;
}

void enkiNBTFreeAllocations( enkiNBTDataStream* pStream_ )
{
	free( pStream_->pAllocation );
	memset( pStream_, 0, sizeof(enkiNBTDataStream) );
}

int8_t  enkiNBTReadInt8( enkiNBTDataStream* pStream_ )
{
	int8_t retVal = pStream_->pCurrPos[ 0 ];
	pStream_->pCurrPos += 1;
	return retVal;
}

int8_t  enkiNBTReadByte( enkiNBTDataStream* pStream_ )
{
	return enkiNBTReadInt8( pStream_ );
}

int16_t enkiNBTReadInt16( enkiNBTDataStream* pStream_ )
{
	int16_t retVal = ( pStream_->pCurrPos[ 0 ] << 8 ) + pStream_->pCurrPos[ 1 ];
	pStream_->pCurrPos += 2;
	return retVal;
}

int16_t enkiNBTReadShort( enkiNBTDataStream* pStream_ )
{
	return enkiNBTReadInt16( pStream_ );
}

int32_t enkiNBTReadInt32( enkiNBTDataStream* pStream_ )
{
	int32_t retVal = ( pStream_->pCurrPos[ 0 ] << 24 ) + ( pStream_->pCurrPos[ 1 ] << 16 ) + ( pStream_->pCurrPos[ 2 ] << 8 ) + pStream_->pCurrPos[ 3 ];
	pStream_->pCurrPos += 4;
	return retVal;
}

int32_t enkiNBTReadInt( enkiNBTDataStream* pStream_ )
{
	return enkiNBTReadInt32( pStream_ );
}

float   enkiNBTReadFloat( enkiNBTDataStream* pStream_ )
{
	int32_t iVal = ( pStream_->pCurrPos[ 0 ] << 24 ) + ( pStream_->pCurrPos[ 1 ] << 16 ) + ( pStream_->pCurrPos[ 2 ] << 8 ) + pStream_->pCurrPos[ 3 ];
	float retVal = *( float* )&iVal;
	pStream_->pCurrPos += 4;
	return retVal;
}

int64_t enkiNBTReadInt64( enkiNBTDataStream* pStream_ )
{
	int64_t retVal = ( ( int64_t )pStream_->pCurrPos[ 0 ] << 54 ) + ( ( int64_t )pStream_->pCurrPos[ 1 ] << 48 ) + ( ( int64_t )pStream_->pCurrPos[ 2 ] << 40 ) + ( ( int64_t )pStream_->pCurrPos[ 5 ] << 32 ) + 
		             ( pStream_->pCurrPos[ 4 ] << 24 ) + ( pStream_->pCurrPos[ 5 ] << 16 ) + ( pStream_->pCurrPos[ 6 ] << 8 ) + pStream_->pCurrPos[ 7 ];
	pStream_->pCurrPos += 8;
	return retVal;
}

int64_t enkiNBTReadlong( enkiNBTDataStream* pStream_ )
{
	return enkiNBTReadInt64( pStream_ );
}

double  enkiNBTReadDouble( enkiNBTDataStream* pStream_ )
{
	int64_t iVal = ( ( int64_t )pStream_->pCurrPos[ 0 ] << 54 ) + ( ( int64_t )pStream_->pCurrPos[ 1 ] << 48 ) + ( ( int64_t )pStream_->pCurrPos[ 2 ] << 40 ) + ( ( int64_t )pStream_->pCurrPos[ 5 ] << 32 ) + 
		           ( pStream_->pCurrPos[ 4 ] << 24 ) + ( pStream_->pCurrPos[ 5 ] << 16 ) + ( pStream_->pCurrPos[ 6 ] << 8 ) + pStream_->pCurrPos[ 7 ];
	double retVal = *( double* )&iVal;
	pStream_->pCurrPos += 8;
	return retVal;
}

enkiNBTString enkiNBTReadString( enkiNBTDataStream* pStream_ )
{
	enkiNBTString nbtString;
	nbtString.size = enkiNBTReadInt16( pStream_ );
	nbtString.pStrNotNullTerminated = (const char*)pStream_->pCurrPos;
	return nbtString;
}

static void SkipDataToNextTag( enkiNBTDataStream* pStream_ )
{
	switch( pStream_->currentTag.tagId )
	{
	case enkiNBTTAG_End:
		// no data, so do nothing.
		break;
	case enkiNBTTAG_Byte:
		pStream_->pNextTag += 1; // 1 byte
		break;
	case enkiNBTTAG_Short:
		pStream_->pNextTag += 2; // 2 bytes
		break;
	case enkiNBTTAG_Int:
		pStream_->pNextTag += 4;
		break;
	case enkiNBTTAG_Long:
		pStream_->pNextTag += 8;
		break;
	case enkiNBTTAG_Float:
		pStream_->pNextTag += 4;
		break;
	case enkiNBTTAG_Double:
		pStream_->pNextTag += 8;
		break;
	case enkiNBTTAG_Byte_Array:
	{
		int32_t length = enkiNBTReadInt32( pStream_ );
		pStream_->pNextTag = pStream_->pCurrPos + length * 1; // array of bytes
		break;
	}
	case enkiNBTTAG_String:
	{
		int32_t length = enkiNBTReadInt16( pStream_ );
		pStream_->pNextTag = pStream_->pCurrPos + length;
		break;
	}
	case enkiNBTTAG_List:
		// read as a compound type
		break;
	case enkiNBTTAG_Compound:
		// data is in standard format, so do nothing.
		break;
	case enkiNBTTAG_Int_Array:
	{
		int32_t length = enkiNBTReadInt32( pStream_ );
		pStream_->pNextTag = pStream_->pCurrPos + length * 4; // array of ints (4 bytes)
		break;
	}
	default:
		assert( 0 );
		break;
	}
}

int enkiNBTReadNextTag( enkiNBTDataStream* pStream_ )
{
	if( ( enkiNBTTAG_Compound == pStream_->currentTag.tagId ) || ( enkiNBTTAG_List == pStream_->currentTag.tagId ) )
	{
		pStream_->level++;
		pStream_->parentTags[ pStream_->level ] = pStream_->currentTag;
	}
	if( ( pStream_->level >= 0 ) && ( enkiNBTTAG_List == pStream_->parentTags[ pStream_->level ].tagId ) )
	{
		if( pStream_->parentTags[ pStream_->level ].listCurrItem == pStream_->parentTags[ pStream_->level ].listNumItems )
		{
			pStream_->level--;
		}
		else
		{
			pStream_->currentTag.tagId = pStream_->parentTags[ pStream_->level ].listItemTagId;
			pStream_->currentTag.pName = NULL;
			SkipDataToNextTag( pStream_ );
			pStream_->parentTags[ pStream_->level ].listCurrItem++;
			return 1;
		}
	}

	if( pStream_->pNextTag >= pStream_->pDataEnd )
	{
		return 0;
	}
	pStream_->pCurrPos = pStream_->pNextTag;

	// Get Tag Header
	pStream_->currentTag.pName = NULL;
	pStream_->currentTag.tagId = *(pStream_->pCurrPos++);
	if( enkiNBTTAG_End != pStream_->currentTag.tagId )
	{
		if( 0xff == *(pStream_->pCurrPos) )
		{
			pStream_->pCurrPos++;
			pStream_->currentTag.pName = ( char* )pStream_->pCurrPos;
			while( *(pStream_->pCurrPos++) != 0 );
		}
		else
		{
			int32_t lengthOfName = enkiNBTReadInt16( pStream_ );
			if( lengthOfName )
			{
				*( pStream_->pCurrPos - 2 ) = 0xff; // this value will not be seen as a length since it will be negative
				pStream_->currentTag.pName = ( char* )( pStream_->pCurrPos - 1 );
				memmove( pStream_->currentTag.pName, pStream_->pCurrPos, lengthOfName );
				pStream_->pCurrPos += lengthOfName - 1;
				pStream_->pCurrPos[ 0 ] = 0; // null terminator
				pStream_->pCurrPos += 1;
			}
		}
	}
	if( enkiNBTTAG_List == pStream_->currentTag.tagId )
	{
		pStream_->currentTag.listItemTagId = *(pStream_->pCurrPos++);
		pStream_->currentTag.listNumItems = enkiNBTReadInt32( pStream_ );
		pStream_->currentTag.listCurrItem = 0;
	}
	pStream_->pNextTag = pStream_->pCurrPos;

	SkipDataToNextTag( pStream_ );

	if( ( pStream_->level >= 0 ) && ( enkiNBTTAG_End == pStream_->currentTag.tagId ) )
	{
		pStream_->level--;
	}

	return 1;
}


void enkiNBTRewind( enkiNBTDataStream* pStream_ )
{
	memset( &(pStream_->currentTag), 0, sizeof( enkiNBTTagHeader ) );
	pStream_->pCurrPos = pStream_->pData;
	pStream_->level = -1;
	pStream_->pNextTag = pStream_->pData;
}

void enkiRegionFileInit( enkiRegionFile* pRegionFile_ )
{
	memset( pRegionFile_, 0, sizeof( enkiRegionFile ) );
}


enkiRegionFile enkiRegionFileLoad( FILE * fp_ )
{
	enkiRegionFile regionFile;
	enkiRegionFileInit( &regionFile );
	fseek( fp_, 0, SEEK_END );
	regionFile.regionDataSize = ftell( fp_ );
	fseek( fp_, 0, SEEK_SET ); // return to start position

	// get the data in the chunks data section
	regionFile.pRegionData = (uint8_t*)malloc( regionFile.regionDataSize );
	fread( regionFile.pRegionData, 1, regionFile.regionDataSize, fp_ ); // note: because sectionDataChunks is an array of single bytes, sizeof( sectionDataChunks ) == sectionDataSize

	return regionFile;
}


void enkiInitNBTDataStreamForChunk( enkiRegionFile regionFile_, int32_t chunkNr_, enkiNBTDataStream* pStream_ )
{
	RegionHeader* header = (RegionHeader*)regionFile_.pRegionData;
	uint32_t locationOffset = GetChunkLocation( header->sectionChunksInfos[ chunkNr_ ] );
	if( locationOffset > sizeof( RegionHeader ) )
	{
		uint32_t length = Get32BitInt(  *( BigEndian4BytesTo32BitInt* )&regionFile_.pRegionData[ locationOffset ] );
		uint8_t compression_type = regionFile_.pRegionData[ locationOffset + 4 ]; // we ignore this as unused for now
		--length; // length includes compression_type
		// get the data and decompress it
		uint8_t* dataCompressed = &regionFile_.pRegionData[ locationOffset + 5 ];
		enkiNBTInitFromMemoryCompressed( pStream_, dataCompressed, length, 0 );
	}
	else
	{
		enkiNBTInitFromMemoryUncompressed( pStream_, NULL, 0 ); // clears stream
	}
}

int32_t enkiGetTimestampForChunk( enkiRegionFile regionFile_, int32_t chunkNr_ )
{
	RegionHeader* header = (RegionHeader*)regionFile_.pRegionData;
	return Get32BitInt( header->sectionChunksTimestamps[ chunkNr_ ] );
}

void enkiRegionFileFreeAllocations(enkiRegionFile * pRegionFile_)
{
	free( pRegionFile_->pRegionData );
	memset( pRegionFile_, 0, sizeof(enkiRegionFile) );
}

int enkiAreStringsEqual( const char * lhs_, const char * rhs_ )
{
	if( lhs_ && rhs_ )
	{
		if( 0 == strcmp( lhs_, rhs_ ) )
		{
			return 1;
		}
	}
	return 0;
}

void enkiChunkInit( enkiChunkBlockData* pChunk_ )
{
	memset( pChunk_, 0, sizeof( enkiChunkBlockData ) );
}

enkiChunkBlockData enkiNBTReadChunk( enkiNBTDataStream * pStream_ )
{
	enkiChunkBlockData chunk;
	enkiChunkInit( &chunk );
	while( enkiNBTReadNextTag( pStream_ ) )
	{
		if( enkiAreStringsEqual( "Level", pStream_->currentTag.pName ) )
		{
			int foundXPos = 0;
			int foundZPos = 0;
			int foundSection = 0;
			while( enkiNBTReadNextTag( pStream_ ) )
			{
				if( enkiAreStringsEqual( "xPos", pStream_->currentTag.pName ) )
				{
					foundXPos = 1;
					chunk.xPos = enkiNBTReadInt32( pStream_ );
				}
				else if( enkiAreStringsEqual( "zPos", pStream_->currentTag.pName ) )
				{
					foundZPos = 1;
					chunk.zPos = enkiNBTReadInt32( pStream_ );
				}
				else if( enkiAreStringsEqual( "Sections", pStream_->currentTag.pName ) )
				{
					foundSection = 1;
					int32_t levelParent = pStream_->level;
					int8_t sectionY = -1;
					uint8_t* pBlocks = NULL;
					do
					{
						if( 0 == enkiNBTReadNextTag( pStream_ ) )
						{
							break;
						}
						if( enkiNBTTAG_Compound == pStream_->currentTag.tagId )
						{
							chunk.countOfSections++;
						}
						if( enkiAreStringsEqual( "Blocks", pStream_->currentTag.pName ) )
						{
							pBlocks = pStream_->pCurrPos;
						}
						if( enkiAreStringsEqual( "Y", pStream_->currentTag.pName ) )
						{
							sectionY = enkiNBTReadInt8( pStream_ );
						}
						if( pBlocks && ( 0 <= sectionY ) )
						{
							chunk.sections[ sectionY ] = pBlocks;
							sectionY = -1;
							pBlocks = NULL;
						}
					} while(  pStream_->level > levelParent );
				}

				if( foundXPos && foundZPos && foundSection )
				{
					return chunk;
				}
			}
		}
	}
	// reset to empty
	enkiChunkInit( &chunk );
	return chunk;
}

enkiMICoordinate enkiGetChunkOrigin(enkiChunkBlockData * pChunk_)
{
	enkiMICoordinate retVal;
	retVal.x = pChunk_->xPos * ENKI_MI_SIZE_SECTIONS;
	retVal.y = 0;
	retVal.z = pChunk_->zPos * ENKI_MI_SIZE_SECTIONS;
	return retVal;
}

enkiMICoordinate enkiGetChunkSectionOrigin(enkiChunkBlockData * pChunk_, int32_t section_)
{
	enkiMICoordinate retVal;
	retVal.x = pChunk_->xPos * ENKI_MI_SIZE_SECTIONS;
	retVal.y = section_ * ENKI_MI_SIZE_SECTIONS;
	retVal.z = pChunk_->zPos * ENKI_MI_SIZE_SECTIONS;
	return retVal;
}

uint8_t enkiGetChunkSectionVoxel(enkiChunkBlockData * pChunk_, int32_t section_, enkiMICoordinate sectionOffset_)
{
	uint8_t retVal = 0;
	uint8_t* pSection = pChunk_->sections[ section_ ];
	uint8_t* pVoxel = pSection + sectionOffset_.y*ENKI_MI_SIZE_SECTIONS*ENKI_MI_SIZE_SECTIONS + sectionOffset_.z*ENKI_MI_SIZE_SECTIONS + sectionOffset_.x;
	retVal = *pVoxel;
	return retVal;
}

uint32_t* enkiGetMineCraftPalette()
{
	return minecraftPalette;
}
