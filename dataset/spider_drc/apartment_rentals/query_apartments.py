"""
Simple SQL Query Executor for apartment_rentals.sqlite

This script executes SQL queries directly on the SQLite database.
"""

import sqlite3
from pathlib import Path


def execute_query(db_path: str, query: str):
    """
    Execute a SQL query on a SQLite database and return results.
    
    Args:
        db_path: Path to the SQLite database file
        query: SQL query string to execute
        
    Returns:
        List of tuples containing query results
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        conn.close()
        return results, column_names
    except sqlite3.Error as e:
        print(f"Error executing query: {e}")
        return [], []


def print_results(results, column_names, query):
    """Print query results in a readable format."""
    print(f"\nQuery:\n{query}\n")
    print("=" * 80)
    
    if not results:
        print("No results found.")
        return
    
    # Print column headers
    print(" | ".join(column_names))
    print("-" * 80)
    
    # Print rows
    for row in results:
        print(" | ".join(str(val) for val in row))
    
    print(f"\nTotal rows: {len(results)}\n")


def main():
    """Main function."""
    db_path = Path(__file__).parent / "apartment_rentals.sqlite"
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    # Retrieve all apartments that have at least 2 bedrooms and are located in a building managed by Jennifer.
    query1 = """
    SELECT *
    FROM Apartments A
    JOIN Apartment_Buildings B
      ON A.building_short_name = B.building_short_name
    WHERE A.bedroom_count >= 2
      AND B.building_manager = 'Jennifer';
    """

    # Retrieve all apartments that include a gym facility and are currently available, along with their facility code and availability status.
    query2 = """
    SELECT distinct A.apt_number,
      A.apt_type_code,
      F.facility_code,
      V.available_yn
    FROM Apartments A
    JOIN Apartment_Facilities F
      ON A.apt_number = F.apt_number
    JOIN View_Unit_Status V
      ON A.apt_number = V.apt_number
    WHERE F.facility_code = 'Gym'
      AND V.available_yn = 1;
    """

    # Retrieve all apartments located in buildings managed by Jennifer, along with their booking status and booking dates.
    query3 = """
    SELECT A.apt_number,
      A.bedroom_count,
      A.bathroom_count,
      B.building_manager,
      BK.booking_status_code,
      BK.booking_start_date,
      BK.booking_end_date
    FROM Apartments A
    JOIN Apartment_Buildings B
      ON A.building_short_name = B.building_short_name
    JOIN Apartment_Bookings BK
      ON A.apt_number = BK.apt_number
    WHERE B.building_manager = 'Jennifer';
    """

    # Retrieve all apartments that have at least 5 bedrooms, are located in buildings managed by Jennifer, and include the facility Broadband.
    query4 = """
    SELECT A.apt_number,
      A.bedroom_count,
      B.building_manager,
      F.facility_code
    FROM Apartments A
    JOIN Apartment_Buildings B
      ON A.building_short_name = B.building_short_name
    JOIN Apartment_Facilities F
      ON A.apt_number = F.apt_number
    WHERE B.building_manager = 'Jennifer'
      AND F.facility_code = 'Boardband'
      AND A.bedroom_count >= 5;
    """

    # Retrieve each building manager and the total number of apartments they manage, but only include managers who are responsible for at least 3 apartments.
    query5 = """
    SELECT B.building_manager,
      COUNT(*) AS apartment_count
    FROM Apartments A
    JOIN Apartment_Buildings B
      ON A.building_short_name = B.building_short_name
    GROUP BY B.building_manager
    HAVING COUNT(*) >= 3;
    """

    # Retrieve the average number of bedrooms for each apartment type.
    query6 = """
    SELECT A.apt_type_code,
      AVG(A.bedroom_count) AS avg_bedrooms
    FROM Apartments A
    GROUP BY A.apt_type_code;
    """

    # Retrieve each building's short name along with the total number of bookings for apartments in that building.
    query7 = """
    SELECT B.building_short_name,
      COUNT(*) AS booking_count
    FROM Apartment_Bookings BK
    JOIN Apartments A
      ON BK.apt_number = A.apt_number
    JOIN Apartment_Buildings B
      ON A.building_short_name = B.building_short_name
    GROUP BY B.building_short_name;
    """

    # Retrieve all apartments that do not have any bookings.
    query8 = """
    SELECT A.*
    FROM Apartments A
    WHERE NOT EXISTS (
        SELECT 1
        FROM Apartment_Bookings BK
        WHERE BK.apt_number = A.apt_number
    );
    """

    # Retrieve all apartments that do not have a gym facility.
    query9 = """
    SELECT A.*
    FROM Apartments A
    WHERE A.apt_number NOT IN (
        SELECT F.apt_number
        FROM Apartment_Facilities F
        WHERE F.facility_code = 'Gym'
    );
    """

    query = query5
    results, column_names = execute_query(str(db_path), query)
    print_results(results, column_names, query)


if __name__ == "__main__":
    main()

