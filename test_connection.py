# file: test_connection.py

import asyncio
import asyncmy # We'll test with the driver you have installed.

async def main():
    """
    A minimal script to test the raw database connection,
    bypassing SQLAlchemy and all other project files.
    """
    print("Attempting to connect to MySQL using asyncmy driver...")
    conn = None
    try:
        # Use the exact same credentials. Note the password is a plain string.
        conn = await asyncmy.connect(
            host='localhost',
            port=3306,
            user='root',
            password='Watch%%402025', # Using the literal password
            db='project_db'
        )
        
        cursor = await conn.cursor()
        await cursor.execute("SELECT 1")
        result = await cursor.fetchone()
        
        if result[0] == 1:
            print("\n✅✅✅ SUCCESS! ✅✅✅")
            print("The raw database driver connected successfully.")
            print("This means the problem is likely in the SQLAlchemy connection string or configuration.")
        else:
            print("\n❌ FAILED: Connection succeeded but query failed.")

    except asyncmy.errors.OperationalError as e:
        print("\n❌❌❌ FAILURE ❌❌❌")
        print("The raw database driver failed to connect.")
        print(f"Error: {e}")
        print("This confirms the issue is at a very low level, related to the driver's interaction with MySQL.")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    finally:
        if conn:
            await conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    asyncio.run(main())