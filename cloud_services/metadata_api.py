"""
Metadata API service for Littoral Shoreline Data.

Provides REST API endpoints to query shoreline metadata from BigQuery
and serve public GeoJSON data URLs.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, Response
from google.cloud import bigquery
import pandas as pd

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery configuration
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'useful-theory-442820-q8')
DATASET_ID = 'shoreline_metadata'
TABLE_ID = 'shoreline_data'

# Initialize BigQuery client
bq_client = bigquery.Client(project=PROJECT_ID)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/api/sites', methods=['GET'])
def get_sites():
    """Get list of available sites."""
    try:
        query = f"""
        SELECT 
            site_name,
            COUNT(*) as shoreline_count,
            MIN(image_date) as earliest_date,
            MAX(image_date) as latest_date,
            AVG(total_length_m) as avg_length_m,
            AVG(quality_score) as avg_quality
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE image_date IS NOT NULL
        GROUP BY site_name
        ORDER BY site_name
        """
        
        query_job = bq_client.query(query)
        results = query_job.result()
        
        sites = []
        for row in results:
            sites.append({
                'site_name': row.site_name,
                'shoreline_count': row.shoreline_count,
                'date_range': {
                    'earliest': row.earliest_date.isoformat() if row.earliest_date else None,
                    'latest': row.latest_date.isoformat() if row.latest_date else None
                },
                'avg_length_m': float(row.avg_length_m) if row.avg_length_m else 0,
                'avg_quality': float(row.avg_quality) if row.avg_quality else 0
            })
        
        return jsonify({
            'sites': sites,
            'total_sites': len(sites)
        })
        
    except Exception as e:
        logger.error(f"Error fetching sites: {str(e)}")
        return jsonify({'error': 'Failed to fetch sites'}), 500


@app.route('/api/shorelines', methods=['GET'])
def get_shorelines():
    """Get shoreline data with optional filtering."""
    try:
        # Parse query parameters
        site_name = request.args.get('site')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        min_quality = request.args.get('min_quality', type=float)
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Build query
        where_conditions = []
        if site_name:
            where_conditions.append(f"site_name = '{site_name}'")
        if start_date:
            where_conditions.append(f"image_date >= '{start_date}'")
        if end_date:
            where_conditions.append(f"image_date <= '{end_date}'")
        if min_quality is not None:
            where_conditions.append(f"quality_score >= {min_quality}")
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
        SELECT 
            shoreline_id,
            site_name,
            image_date,
            processing_date,
            geojson_url,
            ST_AsGeoJSON(geometry) as bbox_geojson,
            total_length_m,
            num_features,
            avg_feature_length_m,
            quality_score,
            data_source,
            processing_version
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        {where_clause}
        ORDER BY image_date DESC, site_name
        LIMIT {limit} OFFSET {offset}
        """
        
        query_job = bq_client.query(query)
        results = query_job.result()
        
        shorelines = []
        for row in results:
            shoreline = {
                'shoreline_id': row.shoreline_id,
                'site_name': row.site_name,
                'image_date': row.image_date.isoformat() if row.image_date else None,
                'processing_date': row.processing_date,
                'geojson_url': row.geojson_url,
                'bbox': json.loads(row.bbox_geojson) if row.bbox_geojson else None,
                'metrics': {
                    'total_length_m': float(row.total_length_m) if row.total_length_m else 0,
                    'num_features': int(row.num_features) if row.num_features else 0,
                    'avg_feature_length_m': float(row.avg_feature_length_m) if row.avg_feature_length_m else 0,
                    'quality_score': float(row.quality_score) if row.quality_score else 0
                },
                'data_source': row.data_source,
                'processing_version': row.processing_version
            }
            shorelines.append(shoreline)
        
        return jsonify({
            'shorelines': shorelines,
            'pagination': {
                'limit': limit,
                'offset': offset,
                'count': len(shorelines)
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching shorelines: {str(e)}")
        return jsonify({'error': 'Failed to fetch shorelines'}), 500


@app.route('/api/shorelines/<shoreline_id>', methods=['GET'])
def get_shoreline_detail(shoreline_id: str):
    """Get detailed information for a specific shoreline."""
    try:
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE shoreline_id = '{shoreline_id}'
        """
        
        query_job = bq_client.query(query)
        results = list(query_job.result())
        
        if not results:
            return jsonify({'error': 'Shoreline not found'}), 404
        
        row = results[0]
        
        shoreline_detail = {
            'shoreline_id': row.shoreline_id,
            'site_name': row.site_name,
            'image_date': row.image_date.isoformat() if row.image_date else None,
            'processing_date': row.processing_date,
            'geojson_url': row.geojson_url,
            'bbox': json.loads(row.bbox_geojson) if hasattr(row, 'bbox_geojson') and row.bbox_geojson else None,
            'metrics': {
                'total_length_m': float(row.total_length_m) if row.total_length_m else 0,
                'num_features': int(row.num_features) if row.num_features else 0,
                'avg_feature_length_m': float(row.avg_feature_length_m) if row.avg_feature_length_m else 0,
                'quality_score': float(row.quality_score) if row.quality_score else 0
            },
            'data_source': row.data_source,
            'processing_version': row.processing_version,
            'metadata': json.loads(row.metadata) if row.metadata else {}
        }
        
        return jsonify(shoreline_detail)
        
    except Exception as e:
        logger.error(f"Error fetching shoreline detail: {str(e)}")
        return jsonify({'error': 'Failed to fetch shoreline detail'}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics about the shoreline dataset."""
    try:
        query = f"""
        SELECT 
            COUNT(*) as total_shorelines,
            COUNT(DISTINCT site_name) as total_sites,
            MIN(image_date) as earliest_date,
            MAX(image_date) as latest_date,
            SUM(total_length_m) as total_length_m,
            AVG(quality_score) as avg_quality,
            SUM(num_features) as total_features
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE image_date IS NOT NULL
        """
        
        query_job = bq_client.query(query)
        results = list(query_job.result())
        
        if not results:
            return jsonify({'error': 'No data found'}), 404
        
        row = results[0]
        
        stats = {
            'dataset_summary': {
                'total_shorelines': int(row.total_shorelines) if row.total_shorelines else 0,
                'total_sites': int(row.total_sites) if row.total_sites else 0,
                'date_range': {
                    'earliest': row.earliest_date.isoformat() if row.earliest_date else None,
                    'latest': row.latest_date.isoformat() if row.latest_date else None
                },
                'total_length_km': float(row.total_length_m / 1000) if row.total_length_m else 0,
                'avg_quality': float(row.avg_quality) if row.avg_quality else 0,
                'total_features': int(row.total_features) if row.total_features else 0
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)